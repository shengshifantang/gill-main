from enum import Enum
import subprocess
import sys
import shutil
import torch
import torch.distributed as dist
from torchvision.transforms import functional as F
from torchvision import transforms as T
from transformers import AutoFeatureExtractor
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import requests
from io import BytesIO

import os

def dump_git_status(out_file=sys.stdout, exclude_file_patterns=['*.ipynb', '*.th', '*.sh', '*.txt', '*.json']):
  """Logs git status to stdout."""
  subprocess.call('git rev-parse HEAD', shell=True, stdout=out_file)
  subprocess.call('echo', shell=True, stdout=out_file)
  exclude_string = ''
  subprocess.call('git --no-pager diff -- . {}'.format(exclude_string), shell=True, stdout=out_file)


def get_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img


def truncate_caption(caption: str) -> str:
  """Truncate captions at periods and newlines."""
  if caption is None:
    return ""
  caption = str(caption).strip()
  # 兼容中文句号 "。" 和英文句号 "." 以及换行
  cut_points = []
  for ch in ['\n', '。', '.']:
    idx = caption.find(ch)
    if idx != -1:
      cut_points.append(idx)
  if cut_points:
    caption = caption[: min(cut_points) + 1]
  return caption


def pad_to_size(x, size=256):
  delta_w = size - x.size[0]
  delta_h = size - x.size[1]
  padding = (
    delta_w // 2,
    delta_h // 2,
    delta_w - (delta_w // 2),
    delta_h - (delta_h // 2),
  )
  new_im = ImageOps.expand(x, padding)
  return new_im


class RandCropResize(object):

  """
  Randomly crops, then randomly resizes, then randomly crops again, an image. Mirroring the augmentations from https://arxiv.org/abs/2102.12092
  """

  def __init__(self, target_size):
    self.target_size = target_size

  def __call__(self, img):
    img = pad_to_size(img, self.target_size)
    d_min = min(img.size)
    img = T.RandomCrop(size=d_min)(img)
    t_min = min(d_min, round(9 / 8 * self.target_size))
    t_max = min(d_min, round(12 / 8 * self.target_size))
    t = random.randint(t_min, t_max + 1)
    img = T.Resize(t)(img)
    if min(img.size) < 256:
      img = T.Resize(256)(img)
    return T.RandomCrop(size=self.target_size)(img)


class SquarePad(object):
  """Pads image to square.
  From https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
  """
  def __call__(self, image):
    max_wh = max(image.size)
    p_left, p_top = [(max_wh - s) // 2 for s in image.size]
    p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
    padding = (p_left, p_top, p_right, p_bottom)
    return F.pad(image, padding, 0, 'constant')


def create_image_of_text(text: str, width: int = 224, nrows: int = 2, color=(255, 255, 255), font=None) -> torch.Tensor:
  """Creates a (3, nrows * 14, width) image of text.
  
  支持中文字体显示。如果未提供 font，会尝试加载中文字体。

  Returns:
    cap_img: (3, 14 * nrows, width) image of wrapped text.
  """
  # 确保 text 是字符串类型
  if isinstance(text, bytes):
    # 如果是 bytes，使用 decode 转换为字符串
    try:
      text = text.decode('utf-8', errors='ignore')
    except:
      try:
        text = text.decode('latin-1', errors='ignore')
      except:
        text = ""
  elif not isinstance(text, str):
    # 如果是其他类型，转换为字符串
    text = str(text) if text is not None else ""
  elif text is None:
    text = ""
  
  # 处理特殊情况：如果 text 是 bytes 的字符串表示（如 "b'...'"），尝试提取实际内容
  if isinstance(text, str) and text.startswith("b'") and text.endswith("'") and len(text) > 3:
    try:
      # 尝试解析 bytes 字符串表示
      import ast
      parsed = ast.literal_eval(text)
      if isinstance(parsed, bytes):
        text = parsed.decode('utf-8', errors='ignore')
    except:
      pass
  
  # 如果 text 是空字符串，使用占位符
  if not text or len(text.strip()) == 0:
    text = " "
  
  height = 12
  padding = 5
  effective_width = width - 2 * padding
  
  # 如果没有提供字体，尝试加载中文字体
  if font is None:
    # 尝试常见的中文字体路径
    chinese_font_paths = [
      '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # WQY MicroHei
      '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',     # WQY ZenHei
      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # DejaVu (支持部分中文)
      '/System/Library/Fonts/PingFang.ttc',                # macOS
      'C:/Windows/Fonts/msyh.ttc',                        # Windows 微软雅黑
      'C:/Windows/Fonts/simhei.ttf',                      # Windows 黑体
    ]
    
    font = ImageFont.load_default()  # 默认字体
    for font_path in chinese_font_paths:
      if os.path.exists(font_path):
        try:
          font = ImageFont.truetype(font_path, size=10)
          break
        except:
          continue
  
  # Create a black image to draw text on.
  cap_img = Image.new('RGB', (effective_width * nrows, height), color = (0, 0, 0))
  draw = ImageDraw.Draw(cap_img)
  
  # 最终检查：确保 text 是字符串
  if not isinstance(text, str):
    text = str(text) if text is not None else " "
  
  try:
    # 使用关键字参数确保参数传递正确
    draw.text(xy=(0, 0), text=text, fill=color, font=font)
  except Exception as e:
    # 如果绘制失败，使用占位符
    try:
      draw.text(xy=(0, 0), text=" ", fill=color, font=font)
    except:
      pass  # 如果连占位符都失败，就跳过
  cap_img = F.convert_image_dtype(F.pil_to_tensor(cap_img), torch.float32)  # (3, height, W * nrows)
  cap_img = torch.split(cap_img, effective_width, dim=-1)  # List of nrow elements of shape (3, height, W)
  cap_img = torch.cat(cap_img, dim=1)  # (3, height * nrows, W)
  # Add zero padding.
  cap_img = torch.nn.functional.pad(cap_img, [padding, padding, 0, padding])
  return cap_img


def get_feature_extractor_for_model(model_name: str, image_size: int = 224, train: bool = True):
  if model_name is None or str(model_name).strip().lower() in {"", "none", "null"}:
    print("Skipping feature extractor (visual encoder disabled).")
    return None
  print(f'Using HuggingFace AutoFeatureExtractor for {model_name}.')
  offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
  # Check if model_name is a local path
  # ????????????????????
  is_local_path = False
  if model_name.startswith('./') or model_name.startswith('../') or os.path.isabs(model_name):
    abs_path = os.path.abspath(model_name) if not os.path.isabs(model_name) else model_name
    is_local_path = os.path.exists(abs_path) and os.path.isdir(abs_path)
  # Hub ???? 'openai/clip-vit-large-patch14'???????

  # Prefer local mirrors under ./model when possible
  if not is_local_path:
    local_root = os.environ.get("GILL_LOCAL_MODEL_DIR", "./model")
    base_name = model_name.split("/")[-1]
    candidates = [
      os.path.join(local_root, base_name),
      os.path.join(local_root, model_name.replace("/", "-")),
      os.path.join(local_root, model_name.replace("/", os.sep)),
    ]
    for cand in candidates:
      if cand and os.path.isdir(cand):
        print(f"Loading feature extractor from local mirror: {cand}")
        return AutoFeatureExtractor.from_pretrained(cand, local_files_only=True)

  if is_local_path:
    print(f"Loading feature extractor from local path: {model_name}")
    return AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
  try:
    if offline:
      return AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
    return AutoFeatureExtractor.from_pretrained(model_name)
  except OSError:
    if offline:
      print(f"Warning: feature extractor '{model_name}' not found locally in offline mode. "
            "Set --visual-encoder to a local path or disable it.")
      return None
    raise


def get_pixel_values_for_model(feature_extractor, img: Image.Image):
  if feature_extractor is None:
    raise RuntimeError("feature_extractor is None. Provide a local visual encoder or disable vision features.")
  pixel_values = feature_extractor(img.convert('RGB'), return_tensors="pt").pixel_values[0, ...]  # (3, H, W)
  return pixel_values




def save_checkpoint(state, is_best, filename='checkpoint'):
  torch.save(state, filename + '.pth.tar')
  if is_best:
    shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def accuracy(output, target, padding, topk=(1,), is_chinese=False):
  """
  Computes the accuracy over the k top predictions for the specified values of k
  
  Args:
    output: (N, T, vocab_size) predictions
    target: (N, T) target token IDs
    padding: padding token ID (usually -100)
    topk: tuple of k values
    is_chinese: whether this is a Chinese model (for debugging/logging)
  """
  with torch.no_grad():
    maxk = max(topk)
    if output.shape[-1] < maxk:
      print(f"[WARNING] Less than {maxk} predictions available. Using {output.shape[-1]} for topk.")

    maxk = min(maxk, output.shape[-1])
    batch_size = target.size(0)

    # Take topk along the last dimension.
    _, pred = output.topk(maxk, -1, True, True)  # (N, T, topk)

    # 创建 mask：排除 padding tokens
    mask = (target != padding).type(target.dtype)
    
    # 对于中文模型，额外检查 token ID 的有效性
    if is_chinese:
      # 中文 tokenizer 的 token ID 通常在合理范围内
      # 可以添加额外的过滤（可选）
      valid_token_mask = (target >= 0) & (target < output.shape[-1])
      mask = mask * valid_token_mask.type(target.dtype)

    target_expand = target[..., None].expand_as(pred)
    correct = pred.eq(target_expand)
    correct = correct * mask[..., None].expand_as(correct)

    res = []
    for k in topk:
      correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
      total_valid = mask.sum()
      if total_valid > 0:
        res.append(correct_k.mul_(100.0 / total_valid))
      else:
        # 如果没有有效 token，返回 0
        res.append(torch.tensor(0.0, device=output.device))
    return res


def get_params_count(model, max_name_len: int = 60):
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  return params, total_trainable_params, total_nontrainable_params


def get_params_count_str(model, max_name_len: int = 60):
  padding = 70  # Hardcoded depending on desired amount of padding and separators.
  params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  for name, param_count, shape, trainable in params:
    param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  return param_counts_text


class Summary(Enum):
  NONE = 0
  AVERAGE = 1
  SUM = 2
  COUNT = 3


class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))
    
  def display_summary(self):
    entries = [" *"]
    entries += [meter.summary() for meter in self.meters]
    print(' '.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
    self.name = name
    self.fmt = fmt
    self.summary_type = summary_type
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def all_reduce(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    self.sum, self.count = total.tolist()
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)
  
  def summary(self):
    fmtstr = ''
    if self.summary_type is Summary.NONE:
      fmtstr = ''
    elif self.summary_type is Summary.AVERAGE:
      fmtstr = '{name} {avg:.3f}'
    elif self.summary_type is Summary.SUM:
      fmtstr = '{name} {sum:.3f}'
    elif self.summary_type is Summary.COUNT:
      fmtstr = '{name} {count:.3f}'
    else:
      raise ValueError('invalid summary type %r' % self.summary_type)
    
    return fmtstr.format(**self.__dict__)
