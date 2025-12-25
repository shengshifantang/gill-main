"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple, List

import collections
import logging
import os
import random
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset

from gill import utils


# ============================================
# 中文动态提示模板池（适配 DeepSeek-Base 续写风格）
# ============================================
CHINESE_CAPTION_PROMPTS = [
    "这张图片展示了",
    "图中描绘了",
    "可以看到",
    "画面内容为",
    "图片内容是",
    "该图像呈现了",
    "所展示的场景是",
    "从图中可以观察到",
    "这是一张关于",
    "图片中显示的是",
]

# 中文图像生成提示模板（用于训练模型输出 [IMG] token）
CHINESE_GENERATION_PROMPTS = [
    "请生成一张图片",
    "生成图像",
    "创建一张图",
    "画一张",
    "生成一幅",
]

# Caption 最大长度（中文字符数）
MAX_CAPTION_CHARS = 30

# Increase PIL's decompression bomb limit to handle large images
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely (or set to a large value like 1000000000)

# ============================================
# Kolors Text Encoder 支持
# ============================================
KOLORS_TEXT_ENCODER = None
KOLORS_TOKENIZER = None


def load_kolors_text_encoder(kolors_path: str, device: str = 'cuda'):
    """
    加载 Kolors 的 text encoder 用于生成训练标签。
    
    Args:
        kolors_path: Kolors 模型路径
        device: 设备
    
    Returns:
        (tokenizer, text_encoder)
    """
    global KOLORS_TEXT_ENCODER, KOLORS_TOKENIZER
    
    if KOLORS_TEXT_ENCODER is not None:
        return KOLORS_TOKENIZER, KOLORS_TEXT_ENCODER
    
    from transformers import AutoTokenizer, AutoModel
    
    text_encoder_path = os.path.join(kolors_path, 'text_encoder')
    
    print(f"加载 Kolors text encoder: {text_encoder_path}")
    
    # 添加 text_encoder 路径到 sys.path 以支持 ChatGLM 自定义模块
    import sys
    if text_encoder_path not in sys.path:
        sys.path.insert(0, text_encoder_path)
    
    KOLORS_TOKENIZER = AutoTokenizer.from_pretrained(
        text_encoder_path, 
        trust_remote_code=True
    )
    KOLORS_TEXT_ENCODER = AutoModel.from_pretrained(
        text_encoder_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device).eval()
    
    print(f"✓ Kolors text encoder 加载成功")
    return KOLORS_TOKENIZER, KOLORS_TEXT_ENCODER


def encode_text_with_kolors(text: str, tokenizer, text_encoder, 
                            max_length: int = 256, device: str = 'cuda'):
    """
    使用 Kolors text encoder 编码文本。
    
    参考 Kolors 官方 pipeline 的 encode_prompt 方法：
    - 使用 hidden_states[-2]（倒数第二层）而非 last_hidden_state
    - pooled embedding 使用第一个 token 经过投影
    
    Args:
        text: 输入文本
        tokenizer: Kolors tokenizer
        text_encoder: Kolors text encoder
        max_length: 最大序列长度 (Kolors 使用 256)
        device: 设备
    
    Returns:
        (prompt_embeds, pooled_prompt_embeds): 
            prompt_embeds: (1, max_length, 2048)
            pooled_prompt_embeds: (1, 2048)
    """
    with torch.no_grad():
        # ChatGLM tokenizer 兼容性处理：使用 tokenize + convert_tokens_to_ids
        # 避免 tokenizer.encode() 内部调用 _pad() 时的 padding_side 参数问题
        try:
            # 方法 1: 使用 tokenize + convert_tokens_to_ids（最安全）
            text_tokens = tokenizer.tokenize(str(text))
            tokens = tokenizer.convert_tokens_to_ids(text_tokens)
            
            # 添加特殊 token（如果需要）
            if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                tokens = [tokenizer.bos_token_id] + tokens
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                tokens = tokens + [tokenizer.eos_token_id]
        except Exception as e:
            # 回退方案：完全避免使用 encode()，手动构建 token IDs
            # ChatGLM tokenizer 的 encode() 内部会调用 _pad() 并传递 padding_side，导致错误
            try:
                # 方法 2: 使用 tokenizer 的 __call__ 方法，但禁用所有可能导致 padding 的操作
                # 临时设置 padding_side 为 None（如果可能）
                original_padding_side = getattr(tokenizer, 'padding_side', None)
                try:
                    if hasattr(tokenizer, 'padding_side'):
                        tokenizer.padding_side = 'right'  # 设置为有效值
                    
                    # 使用 __call__ 但禁用 padding
                    inputs = tokenizer(str(text), add_special_tokens=True, padding=False, return_tensors=None, truncation=False)
                    
                    # 恢复原始 padding_side
                    if hasattr(tokenizer, 'padding_side') and original_padding_side is not None:
                        tokenizer.padding_side = original_padding_side
                    
                    # 提取 token IDs
                    if isinstance(inputs, list):
                        tokens = inputs
                    elif isinstance(inputs, dict):
                        tokens = inputs.get('input_ids', [])
                        if not isinstance(tokens, list):
                            tokens = tokens.tolist() if hasattr(tokens, 'tolist') else [tokens]
                    else:
                        tokens = []
                except Exception as e2_inner:
                    # 恢复 padding_side
                    if hasattr(tokenizer, 'padding_side') and original_padding_side is not None:
                        tokenizer.padding_side = original_padding_side
                    raise e2_inner
            except Exception as e2:
                # 最后的回退：使用 pad_token_id
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                tokens = [pad_token_id]
        
        # 手动截断和 padding
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            tokens = tokens + [pad_token_id] * (max_length - len(tokens))
        
        input_ids = torch.tensor([tokens], device=device)
        attention_mask = (input_ids != (tokenizer.pad_token_id or 0)).long()
        
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # ========================================
        # 关键：使用 hidden_states[-2]（倒数第二层）
        # 这与 Kolors 官方 pipeline 一致
        # ========================================
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 使用倒数第二层（官方方式）
            hidden_states = outputs.hidden_states[-2]
        elif hasattr(outputs, 'last_hidden_state'):
            # 回退到最后一层
            hidden_states = outputs.last_hidden_state
        else:
            # 尝试直接使用输出
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # ChatGLM 输出形状是 (seq_len, batch, hidden_dim)，需要转置为 (batch, seq_len, hidden_dim)
        if hidden_states.dim() == 3 and hidden_states.shape[1] == 1:
            # 形状是 (seq_len, 1, hidden_dim)，转置为 (1, seq_len, hidden_dim)
            hidden_states = hidden_states.transpose(0, 1)
        
        # 检查是否需要投影（ChatGLM 输出是 4096 维，需要投影到 2048）
        hidden_dim = hidden_states.shape[-1]
        if hidden_dim == 4096:
            # 检查是否有 text_projection 层
            if hasattr(text_encoder, 'text_projection'):
                prompt_embeds = text_encoder.text_projection(hidden_states)
            else:
                # 如果没有投影层，使用前 2048 维（可能不正确，但保证维度匹配）
                # 注意：这种情况应该在验证脚本中检测到
                prompt_embeds = hidden_states[..., :2048]
        else:
            prompt_embeds = hidden_states
        
        # Pad 到 max_length
        seq_len = prompt_embeds.shape[1]
        if seq_len < max_length:
            pad_size = max_length - seq_len
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad_size), value=0)
        elif seq_len > max_length:
            prompt_embeds = prompt_embeds[:, :max_length, :]
        
        # Pooled embedding: 使用第一个 token 的表示
        # 这与 CLIP 的 [CLS] token 类似
        pooled_prompt_embeds = prompt_embeds[:, 0, :]  # (1, 2048)
        
        return prompt_embeds, pooled_prompt_embeds


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args, split: str, tokenizer, precision: str = 'fp32') -> Dataset:
  assert split in ['train', 'val'
    ], 'Expected split to be one of "train" or "val", got {split} instead.'

  # 如果 args 没有 precision 属性，使用参数值
  if not hasattr(args, 'precision'):
    args.precision = precision
  
  dataset_paths = []
  image_data_dirs = []
  train = split == 'train'
  
  # 检查是否使用 Kolors 目标 embedding
  use_kolors_targets = getattr(args, 'use_kolors_targets', False)
  kolors_path = getattr(args, 'kolors_path', './model/Kolors')
  gen_emb_dim = getattr(args, 'gen_emb_dim', 768)

  # 中文提示词数据增强参数（caption 前缀）
  prompt_aug_mode = getattr(args, 'prompt_aug_mode', 'random')
  prompt_aug_prob = float(getattr(args, 'prompt_aug_prob', 1.0))
  prompt_aug_sep = getattr(args, 'prompt_aug_sep', '')
  
  # 检查是否使用布局数据
  layout_mode = getattr(args, 'layout_mode', False)
  layout_json_path = None
  if layout_mode:
    if split == 'train':
      layout_json_path = getattr(args, 'layout_train_json', None)
    else:
      layout_json_path = getattr(args, 'layout_val_json', None)

  # Default configs for datasets.
  # Folder structure should look like:
  if split == 'train':
    if 'cc3m' in args.dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_train.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/training/'))
    elif 'wukong' in args.dataset:
      # 中文WuKong数据集（图片在dataset_dir下，不是image_dir）
      dataset_paths.append(os.path.join(args.dataset_dir, 'wukong_train.tsv'))
      image_data_dirs.append(os.path.join(args.dataset_dir, 'images/'))
    else:
      raise NotImplementedError

  elif split == 'val':
    if 'cc3m' in args.val_dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_val.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/validation'))
    elif 'wukong' in args.val_dataset:
      # 中文WuKong数据集（图片在dataset_dir下，不是image_dir）
      dataset_paths.append(os.path.join(args.dataset_dir, 'wukong_val.tsv'))
      image_data_dirs.append(os.path.join(args.dataset_dir, 'images/'))
    else:
      raise NotImplementedError

    assert len(dataset_paths) == len(image_data_dirs) == 1, (dataset_paths, image_data_dirs)
  else:
    raise NotImplementedError

  if len(dataset_paths) > 1:
    print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
    dataset = torch.utils.data.ConcatDataset([
      CsvDataset(path, image_dir, tokenizer, 'image',
        'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
        num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens,
        use_kolors_targets=use_kolors_targets, kolors_path=kolors_path, gen_emb_dim=gen_emb_dim,
        layout_mode=layout_mode, layout_json_path=layout_json_path,
        prompt_aug_mode=prompt_aug_mode, prompt_aug_prob=prompt_aug_prob, prompt_aug_sep=prompt_aug_sep)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1:
    # WuKong数据集使用'image_path'而不是'image'
    dataset_name = args.dataset if isinstance(args.dataset, str) else args.dataset[0]
    img_key = 'image_path' if 'wukong' in dataset_name.lower() else 'image'
    dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], tokenizer, img_key,
      'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
      image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
      num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens,
      use_kolors_targets=use_kolors_targets, kolors_path=kolors_path, gen_emb_dim=gen_emb_dim,
      layout_mode=layout_mode, layout_json_path=layout_json_path,
      prompt_aug_mode=prompt_aug_mode, prompt_aug_prob=prompt_aug_prob, prompt_aug_sep=prompt_aug_sep)
  else:
    raise ValueError(f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
  return dataset


class CsvDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, tokenizer, img_key,
               caption_key, feature_extractor_model: str,
               train: bool = True, max_len: int = 32, sep="\t", precision: str = 'fp32',
               image_size: int = 224, retrieval_token_idx: List[int] = [-1], gen_token_idx: List[int] = [-1],
               num_tokens: int = 1, num_clip_tokens: int = 1,
               use_kolors_targets: bool = False, kolors_path: str = './model/Kolors',
               gen_emb_dim: int = 768, layout_mode: bool = False, layout_json_path: Optional[str] = None,
               prompt_aug_mode: str = 'random', prompt_aug_prob: float = 1.0, prompt_aug_sep: str = ''):
    logging.debug(f'Loading tsv data from {input_filename}.')
    df = pd.read_csv(input_filename, sep=sep)

    self.base_image_dir = base_image_dir
    self.images = df[img_key].tolist()
    self.captions = df[caption_key].tolist()

    # 确保caption是字符串
    processed_captions = []
    for caption in self.captions:
      if isinstance(caption, list):
        # 如果caption是列表，只取第一个描述
        caption = caption[0] if len(caption) > 0 else ""
      # 确保是字符串
      caption = str(caption).strip()
      processed_captions.append(caption)
    self.captions = processed_captions
    assert len(self.images) == len(self.captions)

    self.feature_extractor_model = feature_extractor_model
    self.feature_extractor = utils.get_feature_extractor_for_model(
      feature_extractor_model, image_size=image_size, train=False)
    self.image_size = image_size

    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision
    self.retrieval_token_idx = retrieval_token_idx
    self.gen_token_idx = gen_token_idx
    self.num_tokens = num_tokens
    self.num_clip_tokens = num_clip_tokens
    self.train = train  # 保存训练/验证模式

    # 中文提示词数据增强（caption 前缀）
    self.prompt_aug_mode = prompt_aug_mode
    self.prompt_aug_prob = float(prompt_aug_prob)
    self.prompt_aug_sep = prompt_aug_sep if prompt_aug_sep is not None else ""
    
    # Kolors 目标 embedding 支持
    self.use_kolors_targets = use_kolors_targets
    self.kolors_path = kolors_path
    self.gen_emb_dim = gen_emb_dim
    self.kolors_tokenizer = None
    self.kolors_text_encoder = None
    
    if use_kolors_targets:
      print(f"📦 启用 Kolors 目标 embedding (dim={gen_emb_dim}, seq_len={num_clip_tokens})")
      # 延迟加载 Kolors text encoder（在第一次使用时加载）

    # 布局数据支持
    self.layout_mode = layout_mode
    self.layout_data = None
    if layout_mode and layout_json_path and os.path.exists(layout_json_path):
      import json
      with open(layout_json_path, 'r', encoding='utf-8') as f:
        layout_data_list = json.load(f)
      # 构建 image_id -> layout_data 的映射
      self.layout_data = {item["image_path"]: item for item in layout_data_list}
      print(f"📐 启用布局模式，加载了 {len(self.layout_data)} 条布局数据")
    elif layout_mode:
      print(f"⚠️ 布局模式已启用，但未找到布局数据文件: {layout_json_path}")

    self.font = None

    logging.debug('Done loading data.')

  def __len__(self):
    return len(self.captions)

  def _truncate_chinese_caption(self, caption: str, max_chars: int) -> str:
    """截断中文 caption 到指定字符数，保留完整语义"""
    # 移除首尾空白
    caption = caption.strip()
    
    # 如果已经足够短，直接返回
    if len(caption) <= max_chars:
      return caption
    
    # 尝试在标点处截断，保持语义完整
    punctuations = ['。', '，', '、', '；', '！', '？', '.', ',', ';', '!', '?']
    truncated = caption[:max_chars]
    
    # 从后向前找最近的标点
    for i in range(len(truncated) - 1, max(0, len(truncated) - 10), -1):
      if truncated[i] in punctuations:
        return truncated[:i]  # 不包含标点
    
    # 没找到标点，直接截断
    return truncated

  def _get_kolors_embedding_from_file(self, image_id: str):
    """
    从预计算的文件加载 Kolors embedding（推荐方式，速度快 3x+）。
    
    Args:
        image_id: 图像文件名
    
    Returns:
        clip_emb: (num_clip_tokens, gen_emb_dim) 目标 embedding
    """
    # 构建 Kolors embedding 路径
    kolors_emb_path = os.path.join(self.base_image_dir, 'kolors_embs', f'{image_id}.npy')
    
    if os.path.exists(kolors_emb_path):
      with open(kolors_emb_path, 'rb') as f:
        clip_emb = np.load(f)  # (256, 2048)
      return clip_emb
    else:
      return None

  def _get_kolors_embedding_realtime(self, caption: str):
    """
    使用 Kolors text encoder 实时获取目标 embedding（备用方式，较慢）。
    
    Args:
        caption: 输入文本
    
    Returns:
        clip_emb: (num_clip_tokens, gen_emb_dim) 目标 embedding
    """
    # 延迟加载 Kolors text encoder
    if self.kolors_tokenizer is None:
      self.kolors_tokenizer, self.kolors_text_encoder = load_kolors_text_encoder(
        self.kolors_path, device='cuda' if torch.cuda.is_available() else 'cpu'
      )
    
    # 编码文本
    prompt_embeds, pooled_embeds = encode_text_with_kolors(
      caption, 
      self.kolors_tokenizer, 
      self.kolors_text_encoder,
      max_length=self.num_clip_tokens,
      device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 返回 numpy 格式 (num_clip_tokens, gen_emb_dim)
    return prompt_embeds.squeeze(0).cpu().numpy()

  def __getitem__(self, idx):
    max_retries = 10
    retry_count = 0
    original_idx = idx
    
    while retry_count < max_retries:
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      caption = str(self.captions[idx])
      clip_l_path = os.path.join(self.base_image_dir, 'clip_embs', str(self.images[idx]) + '.npy')

      try:
        img = Image.open(image_path)
        images = utils.get_pixel_values_for_model(self.feature_extractor, img)

        # ============================================
        # 获取目标 embedding (CLIP 或 Kolors)
        # ============================================
        image_id = str(self.images[idx])
        
        if self.use_kolors_targets:
          # 优先从预计算文件加载 Kolors embedding（速度快 3x+）
          clip_emb = self._get_kolors_embedding_from_file(image_id)
          
          if clip_emb is None:
            # 回退到实时编码（较慢，但保证可用）
            clip_emb = self._get_kolors_embedding_realtime(caption)
          
          # 验证维度
          assert clip_emb.shape == (self.num_clip_tokens, self.gen_emb_dim), \
            f"Kolors embedding shape mismatch: {clip_emb.shape}, expected ({self.num_clip_tokens}, {self.gen_emb_dim})"
        else:
          # 使用预计算的 CLIP embedding
          with open(clip_l_path, 'rb') as f:
            clip_emb = np.load(f, allow_pickle=True)   # (num_clip_tokens, 768) or (768,)
            # 处理一维embedding（pooled CLIP features）
            if clip_emb.ndim == 1:
              clip_emb = clip_emb.reshape(1, -1)  # (768,) -> (1, 768)
            clip_emb = clip_emb[:self.num_clip_tokens, :]

        # ============================================
        # 中文 Caption 预处理
        # ============================================
        # 1. 截断 caption 到最大长度（保留核心信息）
        caption = self._truncate_chinese_caption(caption, MAX_CAPTION_CHARS)
        
        # 2. 动态选择提示模板（训练时随机，验证时固定）
        prefix = ""
        mode = (self.prompt_aug_mode or "random").lower()
        if mode not in ["none", "fixed", "random"]:
          mode = "random"

        if mode != "none":
          if self.train:
            # 训练时按概率加前缀
            if self.prompt_aug_prob >= 1.0 or random.random() < self.prompt_aug_prob:
              if mode == "fixed":
                prefix = CHINESE_CAPTION_PROMPTS[0]
              else:  # random
                prefix = random.choice(CHINESE_CAPTION_PROMPTS)
          else:
            # 验证时默认固定前缀（保证评估稳定）；如需禁用请用 --prompt-aug-mode none
            prefix = CHINESE_CAPTION_PROMPTS[0]

        # 3. 拼接前缀和 caption（可选分隔符）
        if prefix:
          full_caption = f"{prefix}{self.prompt_aug_sep}{caption}"
        else:
          full_caption = caption
        
        # Generation mode: 添加 [IMG] tokens
        for i in range(self.num_tokens):
          full_caption += f'[IMG{i}]'
        
        tokenized_data = self.tokenizer(
          full_caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        tokens = tokenized_data.input_ids[0]
        caption_len = tokenized_data.attention_mask[0].sum()

        # If IMG tokens are overridden by padding, replace them with the correct token.
        if tokens[-1] not in [self.tokenizer.pad_token_id, self.gen_token_idx[-1]]:
          tokens[-self.num_tokens:] = torch.tensor(self.gen_token_idx).to(dtype=tokens.dtype, device=tokens.device)

        decode_caption = self.tokenizer.decode(tokens, skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        # 直接传入 str，避免 .encode('ascii','ignore') 丢失中文
        cap_img = utils.create_image_of_text(decode_caption, width=self.image_size, nrows=2, font=self.font)

        # 布局数据（如果启用）
        objects = None
        bboxes = None
        if self.layout_mode and self.layout_data is not None:
          layout_item = self.layout_data.get(image_id, None)
          if layout_item:
            objects = layout_item.get("objects", [])
            bboxes = [obj["bbox"] for obj in objects]
            # 转换为 tensor
            if bboxes:
              bboxes = torch.tensor(bboxes, dtype=torch.float32)  # (N, 4)

        # 返回数据（根据是否启用布局模式）
        if self.layout_mode and objects is not None:
          return image_path, images, cap_img, tokens, caption_len, tokens, caption_len, clip_emb, objects, bboxes
        else:
          return image_path, images, cap_img, tokens, caption_len, tokens, caption_len, clip_emb
      except Exception as e:
        retry_count += 1
        if retry_count >= max_retries:
          print(f'ERROR: Failed to load sample after {max_retries} retries. Original idx={original_idx}, last tried={idx}')
          print(f'Last error: {image_path} with caption {caption}: {e}')
          raise RuntimeError(f'Dataset loading failed after {max_retries} retries')
        if retry_count <= 3:  # Only print first few errors to avoid spam
          print(f'Warning: Error reading {image_path}: {e} (retry {retry_count}/{max_retries})')
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)
