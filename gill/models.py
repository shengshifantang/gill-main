from typing import List, Optional
import re
from diffusers import StableDiffusionPipeline
try:
    from diffusers import KolorsPipeline
    KOLORS_AVAILABLE = True
except ImportError:
    KOLORS_AVAILABLE = False
    print("⚠️ KolorsPipeline not available, will use standard StableDiffusionPipeline")
import json
import numpy as np
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
 

from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, OPTForCausalLM, AutoModelForCausalLM
from gill import utils
from gill import layers

_MEASURE_RE = re.compile(r"^(一|二|三|四|五|六|七|八|九|十|两|几|多|每)?(个|只|条|张|把|台|部|辆|块|片|件|根|位|名|对|双|群)")
_NOISE_RE = re.compile(r"(正在|位于|看着|站在|坐在|躺在|趴在|穿着|拿着|走在|骑着)")


def _clean_object_name(name: str, max_len: int = 10, min_len: int = 1) -> str:
  if not isinstance(name, str):
    return ""
  name = name.strip()
  if not name:
    return ""
  name = re.sub(r"<[^>]+>", "", name)
  name = re.sub(r"[\"'“”‘’（）()《》【】\[\]{}<>]", "", name)
  name = re.sub(r"[，,。\.、;；:：!?！？~`·•]", "", name)
  name = re.sub(r"\s+", "", name)
  if not name:
    return ""
  name = _MEASURE_RE.sub("", name)
  if not name:
    return ""
  if "的" in name:
    parts = [p for p in name.split("的") if p]
    if parts:
      name = parts[-1]
  if len(name) < min_len or len(name) > max_len:
    return ""
  if _NOISE_RE.search(name):
    return ""
  return name


class GILLArgs:
  freeze_lm: bool = True
  freeze_vm: bool = True
  # 默认使用 Qwen2.5-7B-Instruct 以适配中文指令场景
  opt_version: str = 'Qwen/Qwen2.5-7B-Instruct'
  visual_encoder: str = 'openai/clip-vit-large-patch14'
  n_visual_tokens: int = 1
  task: str = 'captioning'
  ret_emb_dim: Optional[int] = 256
  gen_emb_dim: Optional[int] = 256
  text_emb_layers: List[int] = [-1]
  gen_token_idx: List[int] = [0]
  retrieval_token_idx: List[int] = [0]
  text_fc_mode: str = 'gill_mapper'
  ret_text_fc_mode: str = 'linear'
  num_tokens: int = 8
  num_clip_tokens: int = 77


class GILLModel(nn.Module):
  def __init__(self, tokenizer, args: GILLArgs = GILLArgs(), device_map: Optional[str] = None):
    super().__init__()
    self.tokenizer = tokenizer
    self.feature_extractor = utils.get_feature_extractor_for_model(args.visual_encoder, train=False)
    self.disable_visual = args.visual_encoder is None or str(args.visual_encoder).strip().lower() in {"", "none", "null"}
    
    # 安全获取 image_token（兼容中英文模型）
    if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
      self.image_token = tokenizer.cls_token_id
    elif hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
      self.image_token = tokenizer.bos_token_id
    else:
      # 使用 eos_token 或默认值
      self.image_token = tokenizer.eos_token_id if (hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None) else 0
      print(f"⚠️ Warning: Using {self.image_token} as image_token (no cls_token_id/bos_token_id found)")
    assert args.text_emb_layers != set(args.text_emb_layers), 'text_emb_layers not unique'
    self.args = args
    self.num_tokens = args.num_tokens
    self.num_clip_tokens = args.num_clip_tokens

    # =========================================================
    # 🚨 快速保护: 检测并强制 Kolors / ChatGLM3 维度对齐
    # 如果用户在配置中已将 gen_emb_dim 设为较大值或 num_clip_tokens 很大，
    # 我们认为用户意在使用 Kolors（ChatGLM3）并强制调整到 4096
    self.is_kolors_config = False
    try:
      if (hasattr(self.args, 'gen_emb_dim') and self.args.gen_emb_dim is not None and self.args.gen_emb_dim >= 2048) or (hasattr(self.args, 'num_clip_tokens') and self.args.num_clip_tokens is not None and self.args.num_clip_tokens >= 256):
        self.is_kolors_config = True
    except Exception:
      self.is_kolors_config = False

    if self.is_kolors_config:
      if not hasattr(self.args, 'gen_emb_dim') or self.args.gen_emb_dim != 4096:
        try:
          old = getattr(self.args, 'gen_emb_dim', None)
          print(f"\n⚠️ [CRITICAL WARNING] 检测到 Kolors 配置 (gen_emb_dim={old})，强制将 gen_emb_dim 修改为 4096 以匹配 ChatGLM3 Text Encoder。\n")
        except Exception:
          pass
        self.args.gen_emb_dim = 4096
      # Kolors 通常需要较多的 clip tokens / token length
      if not hasattr(self.args, 'num_clip_tokens') or self.args.num_clip_tokens < 256:
        self.args.num_clip_tokens = 256
        print("⚠️ 调整 num_clip_tokens 为 256（Kolors 兼容）。")

    opt_version = args.opt_version
    visual_encoder = args.visual_encoder
    n_visual_tokens = args.n_visual_tokens
    print(f"Using {opt_version} for the language model.")
    if self.disable_visual:
      print("Visual encoder disabled; skipping vision model load.")
    else:
      print(f"Using {visual_encoder} for the visual model with {n_visual_tokens} visual tokens.")

    print(f"当前 opt_version: {opt_version}")  # 打印opt_version，确认路径是否正确
    is_local_opt = os.path.exists(opt_version) and os.path.isdir(opt_version)

    # 智能检测模型类型
    # 推荐使用 Qwen2.5-7B-Instruct 或 Qwen2.5-14B-Instruct 进行中文适配
    # 优势：指令遵循能力强、128K上下文、开箱即用
    model_type = None
    if 'opt' in opt_version.lower():
        model_type = 'opt'
    elif 'deepseek' in opt_version.lower():
        model_type = 'deepseek'
    elif 'qwen' in opt_version.lower():
        model_type = 'qwen'
        # Qwen2.5 系列特别推荐用于中文任务
        if '2.5' in opt_version.lower() or 'qwen2.5' in opt_version.lower():
            print("✓ 检测到 Qwen2.5 系列模型（推荐用于中文适配）")
    elif 'glm' in opt_version.lower():
        model_type = 'glm'
    else:
        model_type = 'auto'  # 尝试自动加载
    
    print(f"检测到模型类型: {model_type}")
    
    # 处理多GPU device_map
    lm_device_map = None
    max_memory = None
    if device_map is not None:
        if isinstance(device_map, str) and device_map.startswith("cuda") and "," in device_map:
            # 多GPU模式，例如 "cuda:0,1,2"
            gpu_ids = []
            for part in device_map.split(","):
                part = part.strip()
                if ":" in part:
                    idx = int(part.split(":")[1])
                else:
                    idx = int(part)
                gpu_ids.append(idx)
            # 使用 accelerate 的 device_map="auto" 并限制显存
            max_memory = {i: "22GiB" for i in gpu_ids}
            max_memory["cpu"] = "0GiB"  # 禁止 offload 到 CPU
            lm_device_map = "auto"
            print(f"✓ 使用多GPU加载语言模型: GPUs={gpu_ids}, max_memory={max_memory}")
        else:
            lm_device_map = device_map
    
    try:
        if model_type == 'opt':
            print(f"使用OPTForCausalLM加载: {opt_version}")
            load_kwargs = {
                "local_files_only": is_local_opt,
                "low_cpu_mem_usage": True
            }
            if lm_device_map is not None:
                load_kwargs["device_map"] = lm_device_map
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
                load_kwargs["torch_dtype"] = torch.bfloat16
            if is_local_opt:
                self.lm = OPTForCausalLM.from_pretrained(opt_version, **load_kwargs)
            else:
                self.lm = OPTForCausalLM.from_pretrained(opt_version, **load_kwargs)
        else:
            # 使用AutoModelForCausalLM支持更多模型
            print(f"使用AutoModelForCausalLM加载: {opt_version}")
            load_kwargs = {
                "local_files_only": is_local_opt,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16
            }
            if lm_device_map is not None:
                load_kwargs["device_map"] = lm_device_map
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
            if is_local_opt:
                self.lm = AutoModelForCausalLM.from_pretrained(opt_version, **load_kwargs)
            else:
                self.lm = AutoModelForCausalLM.from_pretrained(opt_version, **load_kwargs)
        
        print(f"✓ 语言模型加载成功 ({model_type})")
    except Exception as e:
        print(f"❌ 语言模型加载失败！错误: {str(e)}")
        raise

    print(f"self.lm 是否定义: {hasattr(self, 'lm')}")  # 确认lm已创建
    self.opt_version = opt_version

    if self.args.freeze_lm:
      self.lm.eval()
      print("Freezing the LM.")
      # 关键修改：开启 Gradient Checkpointing 以节省显存
      if hasattr(self.lm, 'gradient_checkpointing_enable'):
          print("Enabling gradient checkpointing for LM (Save Memory!).")
          self.lm.gradient_checkpointing_enable()
          self.lm.enable_input_require_grads() # 配合 checkpointing 需要

      for param in self.lm.parameters():
        param.requires_grad = False
    else:
      self.lm.train()

    self.retrieval_token_idx = args.retrieval_token_idx
    self.gen_token_idx = args.gen_token_idx
    self.lm.resize_token_embeddings(len(tokenizer))

    self.input_embeddings = self.lm.get_input_embeddings()
    # Cache to avoid printing the same prefix log line every step.
    self._prefix_log_cache = set()

    if self.disable_visual:
      self.visual_model = None
      self.visual_model_name = "none"
      hidden_size = 1024
    else:
      print("Restoring pretrained weights for the visual model.")
      # Check if visual_encoder is a local path
      is_local_path = os.path.exists(visual_encoder) and os.path.isdir(visual_encoder)
      if 'clip' in visual_encoder:
        if is_local_path:
          print(f"Loading CLIP model from local path: {visual_encoder}")
          # 检查是否是 Chinese-CLIP
          if 'chinese' in visual_encoder.lower():
            from transformers import ChineseCLIPModel
            full_model = ChineseCLIPModel.from_pretrained(visual_encoder, local_files_only=True, trust_remote_code=True)
            self.visual_model = full_model.vision_model
            print("Loaded Chinese-CLIP vision model")
          else:
            self.visual_model = CLIPVisionModel.from_pretrained(visual_encoder, local_files_only=True)
        else:
          self.visual_model = CLIPVisionModel.from_pretrained(visual_encoder)
      else:
        if is_local_path:
          self.visual_model = AutoModel.from_pretrained(visual_encoder, local_files_only=True)
        else:
          self.visual_model = AutoModel.from_pretrained(visual_encoder)

      if 'clip' in visual_encoder:
        # 获取 hidden_size
        if hasattr(self.visual_model, 'config'):
          config = self.visual_model.config
          if hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
          elif hasattr(config, 'vision_config') and hasattr(config.vision_config, 'hidden_size'):
            hidden_size = config.vision_config.hidden_size
          else:
            # Chinese-CLIP ViT-L-14 默认 hidden_size
            hidden_size = 1024
          print(f"Visual model hidden_size: {hidden_size}")
        else:
          hidden_size = 1024  # 默认值
          print(f"Using default hidden_size: {hidden_size}")
      else:
        raise NotImplementedError

      if self.args.freeze_vm:
        print("Freezing the VM.")
        self.visual_model.eval()
        for param in self.visual_model.parameters():
          param.requires_grad = False
      else:
        self.visual_model.train()

      self.visual_model_name = visual_encoder

    embedding_dim = self.input_embeddings.embedding_dim * self.args.n_visual_tokens
    self.ret_text_hidden_fcs = nn.ModuleList([])
    self.gen_text_hidden_fcs = nn.ModuleList([])

    # 检查是否需要输出 pooled embedding (用于 Kolors)
    output_pooled = self.args.gen_emb_dim >= 2048 and self.args.num_clip_tokens >= 256
    if output_pooled:
      print(f"📦 检测到 Kolors 配置 (gen_emb_dim={self.args.gen_emb_dim}, num_clip_tokens={self.args.num_clip_tokens})")
      print(f"   启用 pooled embedding 输出")

    for layer_idx in self.args.text_emb_layers:
      if (layer_idx == -1 or layer_idx == self.lm.config.num_hidden_layers) and ('bert' not in opt_version):
        if 'opt' in opt_version:  # OPT models
          in_dim = self.lm.config.word_embed_proj_dim
        else:
          # DeepSeek/Qwen/GLM等模型使用hidden_size
          in_dim = self.lm.config.hidden_size

        self.ret_text_hidden_fcs.append(
          layers.TextFcLayer(in_dim, self.args.ret_emb_dim, num_input_tokens=self.args.num_tokens,
                             num_output_tokens=1, mode=self.args.ret_text_fc_mode))
        self.gen_text_hidden_fcs.append(
          layers.TextFcLayer(in_dim, self.args.gen_emb_dim, num_input_tokens=self.args.num_tokens,
                             num_output_tokens=self.args.num_clip_tokens, mode=self.args.text_fc_mode,
                             output_pooled=output_pooled))

      elif layer_idx < self.lm.config.num_hidden_layers:
        self.ret_text_hidden_fcs.append(layers.TextFcLayer(self.lm.config.hidden_size, self.args.ret_emb_dim, num_input_tokens=self.args.num_tokens, num_output_tokens=1, mode=self.args.ret_text_fc_mode))
        self.gen_text_hidden_fcs.append(layers.TextFcLayer(self.lm.config.hidden_size, self.args.gen_emb_dim, num_input_tokens=self.args.num_tokens, num_output_tokens=self.args.num_clip_tokens, mode=self.args.text_fc_mode, output_pooled=output_pooled))
      else:
        raise ValueError(f'Embedding of layer {layer_idx} was requested but model only has {self.lm.config.num_hidden_layers} layers.')

    self.visual_embeddings = nn.Linear(hidden_size, embedding_dim)

    # Retrieval image FC layer.
    self.visual_fc = nn.Linear(hidden_size, self.args.ret_emb_dim)
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


  def get_visual_embs(self, pixel_values: torch.FloatTensor, mode: str = 'captioning'):
    if mode not in ['captioning', 'retrieval', 'generation']:
      raise ValueError(f"mode should be one of ['captioning', 'retrieval', 'generation'], got {mode} instead.")
    if self.visual_model is None:
      raise RuntimeError("Visual encoder is disabled; cannot compute visual embeddings.")

    # Extract visual embeddings from the vision encoder.
    if 'clip' in self.visual_model_name:
      outputs = self.visual_model(pixel_values)
      encoder_outputs = outputs.pooler_output
    else:
      raise NotImplementedError

    # Use the correct fc based on function argument.
    if mode == 'captioning':
      visual_embs = self.visual_embeddings(encoder_outputs)  # (2, D * n_visual_tokens)
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], self.args.n_visual_tokens, -1))
    elif mode == 'retrieval':
      visual_embs = self.visual_fc(encoder_outputs)  # (2, D * n_visual_tokens)
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1))
    elif mode == 'generation':
      visual_embs = torch.zeros((pixel_values.shape[0], 1, 768), device=pixel_values.device)
    else:
      raise NotImplementedError

    return visual_embs


  def train(self, mode=True):
    # DDP Fix: Ensure frozen models stay in eval mode during training
    lm_was_eval = self.args.freeze_lm and not self.lm.training
    vm_was_eval = self.visual_model is not None and self.args.freeze_vm and not self.visual_model.training
    
    super(GILLModel, self).train(mode=mode)
    
    if self.args.freeze_lm and lm_was_eval:
      self.lm.eval()
    if self.visual_model is not None and self.args.freeze_vm and vm_was_eval:
      self.visual_model.eval()


  def forward(
    self,
    pixel_values: torch.FloatTensor,
    labels: Optional[torch.LongTensor] = None,
    caption_len: Optional[torch.LongTensor] = None,
    mode: str = 'captioning',
    concat_captions: bool = False,
    input_prefix: Optional[str] = None,
  ):
    visual_embs = self.get_visual_embs(pixel_values, mode)

    # -------------------------------------------------------------------------
    # DDP 修复：注入 Dummy Gradient 确保所有参数都在计算图中
    # 由于我们在 main.py 中累积所有 mode 的 loss 后只进行一次 backward，
    # 我们需要确保所有参数（包括未使用的）都在计算图中，避免 DDP 报错
    # -------------------------------------------------------------------------
    ddp_dummy_loss = torch.tensor(0.0, dtype=visual_embs.dtype, device=visual_embs.device)
    
    # 根据 mode 注入未使用参数的 dummy gradient
    if mode == 'captioning':
        # Captioning 使用 visual_embeddings，未使用: visual_fc, ret_*, gen_*, logit_scale
        for p in self.visual_fc.parameters():
            ddp_dummy_loss = ddp_dummy_loss + (p.sum() * 0)
        for m in self.ret_text_hidden_fcs:
            for p in m.parameters():
                ddp_dummy_loss = ddp_dummy_loss + (p.sum() * 0)
        for m in self.gen_text_hidden_fcs:
            for p in m.parameters():
                ddp_dummy_loss = ddp_dummy_loss + (p.sum() * 0)
        # logit_scale 只在 retrieval 模式下使用
        ddp_dummy_loss = ddp_dummy_loss + (self.logit_scale.sum() * 0)
            
    elif mode == 'retrieval':
        # Retrieval 使用 visual_fc, ret_*, logit_scale，未使用: visual_embeddings, gen_*
        for p in self.visual_embeddings.parameters():
            ddp_dummy_loss = ddp_dummy_loss + (p.sum() * 0)
        for m in self.gen_text_hidden_fcs:
            for p in m.parameters():
                ddp_dummy_loss = ddp_dummy_loss + (p.sum() * 0)

    elif mode == 'generation':
        # Generation 使用 visual_embeddings, gen_*，未使用: visual_fc, ret_*, logit_scale
        for p in self.visual_fc.parameters():
            ddp_dummy_loss = ddp_dummy_loss + (p.sum() * 0)
        for m in self.ret_text_hidden_fcs:
            for p in m.parameters():
                ddp_dummy_loss = ddp_dummy_loss + (p.sum() * 0)
        # logit_scale 只在 retrieval 模式下使用
        ddp_dummy_loss = ddp_dummy_loss + (self.logit_scale.sum() * 0)

    # 将 dummy loss 加到 visual_embs 上，确保所有参数都在计算图中
    visual_embs = visual_embs + ddp_dummy_loss.expand_as(visual_embs)
    # -------------------------------------------------------------------------

    batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens
    if labels is not None:
      assert labels.shape[0] == batch_size, (visual_embs.shape, labels.shape)
    visual_embs_norm = ((visual_embs ** 2).sum(dim=-1) ** 0.5).mean()

    input_embs = self.input_embeddings(labels)  # (N, T, D)
    input_embs_norm = ((input_embs ** 2).sum(dim=-1) ** 0.5).mean()

    last_embedding_idx = caption_len - 1  # -1 to retrieve the token before the eos token

    if input_prefix is not None:
      prompt_ids = self.tokenizer(input_prefix, add_special_tokens=False, return_tensors="pt").input_ids
      prompt_ids = prompt_ids.to(device=visual_embs.device, dtype=torch.long)  # 保持 long 类型
      prompt_embs = self.input_embeddings(prompt_ids)
      prompt_embs = prompt_embs.repeat(batch_size, 1, 1)
      assert prompt_embs.shape[0] == batch_size, prompt_embs.shape
      assert prompt_embs.shape[2] == input_embs.shape[2], prompt_embs.shape
      assert len(prompt_embs.shape) == 3, prompt_embs.shape

    if mode == 'captioning':
      # Concat to text embeddings.
      condition_seq_len = 0
      if input_prefix is None:
        # Just add visual embeddings.
        input_embs = torch.cat([visual_embs, input_embs], axis=1)
        last_embedding_idx += vis_seq_len
        condition_seq_len += vis_seq_len
        full_labels = torch.zeros(visual_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100
      else:
        cache_key = (mode, False)
        if cache_key not in self._prefix_log_cache:
          print(f'Adding prefix "{input_prefix}" to {mode}.')
          self._prefix_log_cache.add(cache_key)
        # Add visual and prompt embeddings.
        prefix_embs = torch.cat([visual_embs, prompt_embs], axis=1)
        input_embs = torch.cat([prefix_embs, input_embs], axis=1)

        last_embedding_idx += prefix_embs.shape[1]
        condition_seq_len += prefix_embs.shape[1]
        full_labels = torch.zeros(prefix_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100

      # Mask out embedding tokens in the labels.
      full_labels = torch.cat([full_labels, labels], axis=1)

      pad_idx = []
      
      for idx, label in enumerate(full_labels):
        mask_triggered = False
        mask_position = -1
        mask_token = -1
        for k, token in enumerate(label):
          # Mask out retrieval/gen tokens if they exist.
          # 安全获取 pad_token_id（可能为 None）
          pad_token_id = self.tokenizer.pad_token_id if (hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None) else 0
          if token in [pad_token_id] + self.retrieval_token_idx + self.gen_token_idx:
            mask_position = k
            mask_token = token.item()
            label[k:] = -100
            pad_idx.append(k)
            mask_triggered = True
            break
          if k == len(label) - 1:  # No padding found.
            pad_idx.append(k + 1)
        
        # Debug: Only print if there are zero valid labels (problematic case)
        if idx == 0 and not hasattr(self, '_debug_zero_labels_warned'):
          valid_count = (label != -100).sum().item()
          if valid_count == 0:
            self._debug_zero_labels_warned = True
            print(f"\n[WARNING] Zero valid labels detected in captioning mode!")
            print(f"  pad_token_id: {self.tokenizer.pad_token_id}")
            print(f"  Masking triggered at position {mask_position}, token={mask_token}")
            print(f"  First 30 tokens after masking: {label[:30].tolist()}")
            print(f"  This will cause NaN loss!\n")
      
      assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

      bs, seq_len, embs_dim = input_embs.shape
      if concat_captions:
        print('Concatenating examples for captioning!')
        assert len(input_embs.shape) == 3, input_embs
        assert len(full_labels.shape) == 2, full_labels
        assert batch_size % 2 == 0
        all_concat_input_embs = []
        all_concat_labels = []

        # Rearrange embeddings and labels (and their padding) to concatenate captions.
        for i in range(batch_size // 2):
          first_idx = i * 2
          second_idx = first_idx + 1
          first_emb = input_embs[first_idx, :pad_idx[first_idx], :]
          first_labels = full_labels[first_idx, :pad_idx[first_idx]]
          first_padding = input_embs[first_idx, pad_idx[first_idx]:, :]
          first_labels_padding = full_labels[first_idx, pad_idx[first_idx]:]

          second_emb = input_embs[second_idx, :pad_idx[second_idx], :]
          second_labels = full_labels[second_idx, :pad_idx[second_idx]]
          second_padding = input_embs[second_idx, pad_idx[second_idx]:, :]
          second_labels_padding = full_labels[second_idx, pad_idx[second_idx]:]
          bos_idx = visual_embs.shape[1]

          assert torch.all(first_labels_padding == -100), first_labels_padding
          assert torch.all(second_labels_padding == -100), second_labels_padding
          # 安全检查 bos_token（中文模型可能没有）
          if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            assert torch.all(second_labels[bos_idx] == self.tokenizer.bos_token_id), (second_labels, bos_idx, self.tokenizer.bos_token_id)
          else:
            # 如果没有 bos_token，跳过检查（某些中文模型不使用 bos_token）
            print(f"⚠️ Warning: No bos_token_id found, skipping bos_token check")
          
          # Remove BOS token of the second caption.
          second_labels = torch.cat([second_labels[:bos_idx], second_labels[bos_idx + 1:]], axis=0)
          second_emb = torch.cat([second_emb[:bos_idx, :], second_emb[bos_idx + 1:, :]], axis=0)

          concat_input_embs = torch.cat([first_emb, second_emb, first_padding, second_padding], axis=0)   # (T*2, 768)
          concat_labels = torch.cat([first_labels, second_labels, first_labels_padding, second_labels_padding], axis=0)   # (T*2, 768)
          all_concat_input_embs.append(concat_input_embs)
          all_concat_labels.append(concat_labels)

        # Pad to max length.
        input_embs = torch.stack(all_concat_input_embs, axis=0)  # (N/2, T*2, 768)
        full_labels = torch.stack(all_concat_labels, axis=0)  # (N/2, T*2, 768)
        print("Concatenated full_labels:", full_labels[0, ...])
        assert input_embs.shape == (bs // 2, seq_len * 2 - 1, embs_dim), input_embs.shape
        assert full_labels.shape == (bs // 2, seq_len * 2 - 1), full_labels.shape

      output = self.lm(inputs_embeds=input_embs,
                       labels=full_labels,
                       output_hidden_states=True)
    elif mode in ['retrieval', 'generation']:
      full_labels = torch.clone(labels)
      if input_prefix is not None:
        cache_key = (mode, False)
        if cache_key not in self._prefix_log_cache:
          print(f'Adding prefix "{input_prefix}" to {mode}.')
          self._prefix_log_cache.add(cache_key)
        # Add prompt embeddings.
        prefix_embs = prompt_embs
        input_embs = torch.cat([prefix_embs, input_embs], axis=1)
        last_embedding_idx += prefix_embs.shape[1]
        full_labels = torch.cat([
          torch.zeros(prefix_embs.shape[:2], dtype=torch.int64).to(labels.device) - 100,
          full_labels
        ], axis=1)
      
      pad_idx = []
      for label in full_labels:
        for k, token in enumerate(label):
          if (token == self.tokenizer.pad_token_id):
            label[k:] = -100
            pad_idx.append(k)
            break
          if k == len(label) - 1:  # No padding found.
            pad_idx.append(k + 1)
      assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

      bs, seq_len, embs_dim = input_embs.shape
      # Concatenate examples for captioning, if specified.
      if concat_captions:
        print(f'Concatenating examples for {mode}!')
        assert len(input_embs.shape) == 3, input_embs
        assert len(full_labels.shape) == 2, full_labels
        assert batch_size % 2 == 0
        all_concat_input_embs = []
        all_concat_labels = []
        all_last_embedding_idx = []

        # Rearrange embeddings and labels (and their padding) to concatenate captions.
        for i in range(batch_size // 2):
          first_idx = i * 2
          second_idx = first_idx + 1
          first_emb = input_embs[first_idx, :pad_idx[first_idx], :]
          first_labels = full_labels[first_idx, :pad_idx[first_idx]]
          first_padding = input_embs[first_idx, pad_idx[first_idx]:, :]
          first_labels_padding = full_labels[first_idx, pad_idx[first_idx]:]

          second_emb = input_embs[second_idx, :pad_idx[second_idx], :]
          second_labels = full_labels[second_idx, :pad_idx[second_idx]]
          second_padding = input_embs[second_idx, pad_idx[second_idx]:, :]
          second_labels_padding = full_labels[second_idx, pad_idx[second_idx]:]

          bos_idx = 0
          assert torch.all(first_labels_padding == -100), first_labels_padding
          assert torch.all(second_labels_padding == -100), second_labels_padding
          # 安全检查 bos_token（中文模型可能没有）
          if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            assert torch.all(second_labels[bos_idx] == self.tokenizer.bos_token_id), (second_labels, bos_idx, self.tokenizer.bos_token_id)
          else:
            # 如果没有 bos_token，跳过检查（某些中文模型不使用 bos_token）
            print(f"⚠️ Warning: No bos_token_id found, skipping bos_token check")
          
          # Remove BOS token of second caption.
          second_labels = second_labels[bos_idx + 1:]
          second_emb = second_emb[bos_idx + 1:, :]
          last_embedding_idx[second_idx] = last_embedding_idx[second_idx] - 1

          concat_input_embs = torch.cat([first_emb, second_emb, first_padding, second_padding], axis=0)   # (T*2, 768)
          concat_labels = torch.cat([first_labels, second_labels, first_labels_padding, second_labels_padding], axis=0)   # (T*2, 768)
          all_concat_input_embs.append(concat_input_embs)
          all_concat_labels.append(concat_labels)

          all_last_embedding_idx.append((last_embedding_idx[first_idx], first_emb.shape[0] + last_embedding_idx[second_idx]))

          if mode == 'retrieval':
            assert concat_labels[all_last_embedding_idx[-1][0]] in self.retrieval_token_idx, (concat_labels, all_last_embedding_idx[-1][0])
            assert concat_labels[all_last_embedding_idx[-1][1]] in self.retrieval_token_idx, (concat_labels, all_last_embedding_idx[-1][1])
          elif mode == 'generation':
            # Check that the last n tokens are GEN tokens.
            for gen_i in range(len(self.gen_token_idx)):
              assert concat_labels[all_last_embedding_idx[-1][0]-gen_i] == self.gen_token_idx[-gen_i-1], (concat_labels, all_last_embedding_idx[-1][0]-gen_i, self.gen_token_idx[-gen_i-1])
              assert concat_labels[all_last_embedding_idx[-1][1]-gen_i] == self.gen_token_idx[-gen_i-1], (concat_labels, all_last_embedding_idx[-1][1]-gen_i, self.gen_token_idx[-gen_i-1])

        # Pad to max length.
        input_embs = torch.stack(all_concat_input_embs, axis=0)  # (N/2, T*2, 768)
        full_labels = torch.stack(all_concat_labels, axis=0)  # (N/2, T*2, 768)
        assert input_embs.shape == (bs // 2, seq_len * 2 - 1, embs_dim), input_embs.shape
        assert full_labels.shape == (bs // 2, seq_len * 2 - 1), full_labels.shape

      # Update labels to pad non-first tokens.
      for label in full_labels:
        for k, token in enumerate(label):
          if (token == self.tokenizer.pad_token_id) or (token in (self.retrieval_token_idx[1:] + self.gen_token_idx[1:])):
            label[k:] = -100
            break
      output = self.lm(inputs_embeds=input_embs,
                       labels=full_labels,
                       output_hidden_states=True)
    else:
      raise NotImplementedError

    last_embedding = None
    last_output_logit = None
    hidden_states = []
    llm_hidden_states = []

    if mode in ['retrieval', 'generation']:
      num_tokens = self.num_tokens
      if mode == 'retrieval':
        text_hidden_fcs = self.ret_text_hidden_fcs
      else:
        text_hidden_fcs = self.gen_text_hidden_fcs

      # Concatenate captions for retrieval / generation, if specified.
      if not concat_captions:
        for idx, fc_layer in zip(self.args.text_emb_layers, text_hidden_fcs):
          input_hidden_state = torch.stack([output.hidden_states[idx][i, last_embedding_idx[i]-num_tokens+1:last_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
          input_embedding = torch.stack([input_embs[i, last_embedding_idx[i]-num_tokens+1:last_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
          llm_hidden_states.append(input_hidden_state)
          hidden_states.append(fc_layer(input_hidden_state, input_embedding))  # (N, seq_len, 2048)
      else:
        for idx, fc_layer in zip(self.args.text_emb_layers, text_hidden_fcs):
          all_last_embedding = []
          all_input_embedding = []
          all_last_output_logit = []
          for i in range(batch_size // 2):
            first_last_embedding_idx, second_last_embedding_idx = all_last_embedding_idx[i]
            first_last_embedding = output.hidden_states[idx][i, first_last_embedding_idx-num_tokens+1:first_last_embedding_idx+1, :]  # (N, D)
            second_last_embedding = output.hidden_states[idx][i, second_last_embedding_idx-num_tokens+1:second_last_embedding_idx+1, :]  # (N, D)
            all_last_embedding.append(first_last_embedding)
            all_last_embedding.append(second_last_embedding)

            first_input_embs = input_embs[i, first_last_embedding_idx-num_tokens+1:first_last_embedding_idx+1, :]  # (N, D)
            second_input_embs = input_embs[i, second_last_embedding_idx-num_tokens+1:second_last_embedding_idx+1, :]  # (N, D)
            all_input_embedding.append(first_input_embs)
            all_input_embedding.append(second_input_embs)

            first_last_output_logit = output.logits[i, first_last_embedding_idx - 1, :]  # (N, D)
            second_last_output_logit = output.logits[i, second_last_embedding_idx - 1, :]  # (N, D)
            all_last_output_logit.append(first_last_output_logit)
            all_last_output_logit.append(second_last_output_logit)

          last_embedding = torch.stack(all_last_embedding, axis=0)
          input_embedding = torch.stack(all_input_embedding, axis=0)
          last_output_logit = torch.stack(all_last_output_logit, axis=0)
          llm_hidden_states.append(last_embedding)
          hidden_states.append(fc_layer(last_embedding, input_embedding))  # (N, seq_len, 2048)

      if not concat_captions:
        # Add hidden states together.
        last_embedding = torch.stack(hidden_states, dim=-1).sum(dim=-1) #torch.stack([last_hidden_state[i, :, :] for i in range(batch_size)], axis=0)  # (N, T, D)
        last_output_logit = torch.stack([output.logits[i, last_embedding_idx[i] - 1, :] for i in range(batch_size)], axis=0)  # (N, D)
      else:
        # Add hidden states together.
        last_embedding = torch.stack(hidden_states, dim=-1).sum(dim=-1)

      # Compute retrieval loss.
      if mode == 'retrieval':
        assert visual_embs.shape[1] == 1, visual_embs.shape
        assert last_embedding.shape[1] == 1, last_embedding.shape
        visual_embs = visual_embs[:, 0, :]
        visual_embs = visual_embs / visual_embs.norm(dim=1, keepdim=True)
        last_embedding = last_embedding[:, 0, :]
        last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        visual_embs = logit_scale * visual_embs
    elif mode == 'captioning':
      pass
    else:
      raise NotImplementedError

    return output, full_labels, last_embedding, last_output_logit, visual_embs, visual_embs_norm, input_embs_norm, llm_hidden_states

class GILL(nn.Module):
  def __init__(self, tokenizer, model_args: Optional[GILLArgs] = None,
               path_array: Optional[List[str]] = None, emb_matrix: Optional[torch.tensor] = None,
               load_sd: bool = False, num_gen_images: int = 1, decision_model_path: Optional[str] = None, device_map: Optional[str] = None):
    super().__init__()
    # Keep signature for compatibility; unused args are ignored.
    self.model = GILLModel(tokenizer, model_args, device_map=device_map)
    self.load_sd = load_sd
    self.num_gen_images = num_gen_images

    # Load the Stable Diffusion model.
    # 支持指定不同GPU加载图像生成模型
    self.sd_device = os.environ.get('SD_DEVICE', 'cuda:0')  # 默认cuda:0，可通过环境变量指定
    
    # 标记是否使用 Kolors（用于后续生成时的适配）
    self.is_kolors = False
    
    if load_sd:
      # 优先使用Kolors（中文优化），然后是SD v1.5
      kolors_path = "./model/Kolors"
      sd_v15_path = "./model/stable-diffusion-v1-5"
      offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
      
      print(f"📦 图像生成模型将加载到: {self.sd_device}")
      
      if os.path.exists(kolors_path) and KOLORS_AVAILABLE:
        print(f"✓ 加载Kolors模型（中文优化）: {kolors_path}")
        try:
          # 临时禁用 torch.load 安全检查（Kolors 使用 .bin 格式）
          import transformers.modeling_utils as tm_utils
          original_check = getattr(tm_utils, 'check_torch_load_is_safe', None)
          if original_check:
            tm_utils.check_torch_load_is_safe = lambda: None
          
          self.sd_pipe = KolorsPipeline.from_pretrained(
              kolors_path, 
              torch_dtype=torch.float16,
              local_files_only=True,  # 只使用本地文件
              trust_remote_code=True
          ).to(self.sd_device)
          
          # 恢复安全检查
          if original_check:
            tm_utils.check_torch_load_is_safe = original_check
          
          self.is_kolors = True  # 标记使用 Kolors
          print(f"✓ Kolors加载成功！设备: {self.sd_device}")
          print(f"  ⚠️ 注意: Kolors 需要 (B, 256, 2048) 的 prompt_embeds")
          print(f"  当前训练的 gen_emb 维度为 (B, 77, 768)，将使用文本 prompt 生成")
        except Exception as e:
          print(f"⚠️ Kolors加载失败: {e}")
          print("   回退到Stable Diffusion v1.5...")
          if os.path.exists(sd_v15_path):
            model_id = sd_v15_path
          elif offline:
            raise RuntimeError("Kolors failed and offline mode is enabled, but local stable-diffusion-v1-5 not found. Free GPU memory or provide ./model/stable-diffusion-v1-5.")
          else:
            model_id = "runwayml/stable-diffusion-v1-5"
          self.sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(self.sd_device)
      
      elif os.path.exists(sd_v15_path):
        print(f"Loading Stable Diffusion v1.5 from: {sd_v15_path}")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(sd_v15_path, torch_dtype=torch.float16).to("cuda")
      
      else:
        print("Local SD model not found, downloading from HuggingFace...")
        model_id = "runwayml/stable-diffusion-v1-5"
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    # Decision model / retrieval branch removed (not used in current pipeline).

  def __call__(self, images: Tensor, tgt_tokens: Optional[Tensor] = None, caption_len: Optional[Tensor] = None,
               generate: bool = False, num_words: int = 32, temperature: float = 1.0, top_p: float = 1.0,
               ret_scale_factor: float = 1.0, gen_scale_factor: float = 1.0,
               min_word_tokens: int = 0, mode: str = 'captioning', concat_captions: bool = False,
               input_prefix: Optional[str] = None) -> Tensor:
    if generate:
      raise NotImplementedError("GILL image-token generation was removed; use Kolors/SD prompt generation.")
    output = self.model(
      pixel_values = images,
      labels = tgt_tokens,
      caption_len = caption_len,
      mode = mode,
      concat_captions = concat_captions,
      input_prefix = input_prefix)
    return output

  def _generate_text_prompt(self, prompt: str, max_new_tokens: int = 64) -> str:
    """Generate a semantic prompt using the LM only (no GILL mapper)."""
    if self.model is None or self.model.lm is None:
      return prompt
    tokenizer = self.model.tokenizer
    device = self.model.lm.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
      out = self.model.lm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
      )
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

  def _encode_objects(self, object_names: List[str], device: torch.device) -> Optional[torch.Tensor]:
    """
    Encode object names into phrase embeddings for the spatial adapter.
    Returns a tensor of shape (1, N, hidden) in float32 or None if unavailable.
    """
    if not object_names:
      return torch.zeros((1, 0, 4096), device=device, dtype=torch.float32)
    if self.sd_pipe is None or not hasattr(self.sd_pipe, "tokenizer") or self.sd_pipe.tokenizer is None:
      return None
    text_encoder = getattr(self.sd_pipe, "text_encoder", None)
    if text_encoder is None:
      return None

    names = [_clean_object_name(str(n)) for n in object_names]
    hidden = getattr(text_encoder.config, "hidden_size", 4096)
    phrase_emb = torch.zeros((len(names), hidden), device=device, dtype=torch.float32)

    valid_indices = [i for i, n in enumerate(names) if n]
    if not valid_indices:
      return phrase_emb.unsqueeze(0)

    valid_names = [names[i] for i in valid_indices]
    tokenizer = self.sd_pipe.tokenizer
    original_pad = None
    try:
      if tokenizer is not None and hasattr(tokenizer, "_pad"):
        original_pad = tokenizer._pad
        def compatible_pad(encoded_inputs, max_length=None, padding_strategy=None, pad_to_multiple_of=None, return_attention_mask=None, **kwargs):
          kwargs.pop("padding_side", None)
          return original_pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask, **kwargs)
        tokenizer._pad = compatible_pad

      tok_inputs = tokenizer(
        valid_names,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
      ).to(device)
      with torch.no_grad():
        tok_outputs = text_encoder(**tok_inputs)
      attn_mask = tok_inputs.get("attention_mask", None)
      if hasattr(tok_outputs, "last_hidden_state"):
        last_hidden = tok_outputs.last_hidden_state
      elif isinstance(tok_outputs, (tuple, list)) and len(tok_outputs) > 0:
        last_hidden = tok_outputs[0]
      else:
        last_hidden = tok_outputs

      # ChatGLM-style output can be (seq_len, batch, hidden); align to (batch, seq, hidden).
      input_ids = tok_inputs.get("input_ids", None)
      expected_bs = input_ids.shape[0] if input_ids is not None else len(valid_names)
      if last_hidden.dim() == 3 and last_hidden.size(0) != expected_bs and last_hidden.size(1) == expected_bs:
        last_hidden = last_hidden.transpose(0, 1)
      # If batch still mismatches (e.g., duplicated rows), trim to expected size.
      if last_hidden.dim() == 3 and last_hidden.size(0) != expected_bs and last_hidden.size(0) > expected_bs:
        last_hidden = last_hidden[:expected_bs]

      if attn_mask is None or last_hidden.size(0) != attn_mask.size(0):
        pooled = last_hidden.mean(dim=1)
      else:
        # Align seq lengths in case tokenizer/text_encoder mismatch
        seq_len = min(last_hidden.size(1), attn_mask.size(1))
        hs = last_hidden[:, :seq_len, :]
        mask = attn_mask[:, :seq_len].unsqueeze(-1).float()
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
      phrase_emb[valid_indices] = pooled.to(dtype=torch.float32)
    except Exception as e:
      print(f"?? Phrase embedding encoding failed: {e}")
    finally:
      if original_pad is not None and tokenizer is not None:
        tokenizer._pad = original_pad

    return phrase_emb.unsqueeze(0)

  def generate_with_layout(self, prompt: str, enable_layout: bool = True, 
                          enable_feedback: bool = False, layout_planner=None,
                          spatial_adapter=None, feedback_verifier=None,
                          num_words: int = 15, guidance_scale: float = 7.5,
                          num_inference_steps: int = 50, generator=None,
                          scheduled_sampling_ratio: float = 0.4,
                          adapter_scale: float = 1.0,
                          disable_phrase_emb: bool = False,
                          use_gill_prompt: bool = False,
                          max_retries: int = 3,
                          strict_entities: bool = False,
                          enable_cot: bool = False,
                          force_gate: bool = False,
                          gate_value: float = 1.0):
    """
    带布局控制的图像生成（Plan → Generate → Verify 闭环）
    """
    from gill.layout_planner import parse_layout_output
    from gill.spatial_adapter import inject_spatial_control_to_unet, remove_spatial_control_from_unet, SpatialAdapterModuleDict, create_spatial_adapter_for_kolors
    
    result = {
        "image": None,
        "layout": None,
        "feedback": None,
        "semantic_prompt": None,
        "status": None
    }
    history = []
    adapter_container = spatial_adapter
    
    for attempt in range(max_retries + 1):
      current_prompt = prompt
      if attempt > 0 and history and enable_feedback:
        fb_text = history[-1]["feedback"].get("feedback") if history[-1].get("feedback") else None
        if fb_text:
          current_prompt = f"{prompt}\n上一轮反馈: {fb_text}\n请修正布局并重新生成。"
      print(f"🔄 Generation Attempt {attempt + 1}/{max_retries + 1}")
      
      # Step 1: 布局规划
      objects = None
      bboxes = None
      object_names = None
      layout_result = None
      if enable_layout and layout_planner is not None:
        layout_result = layout_planner.generate_layout(
          current_prompt,
          apply_refinement=True,
          strict_entities=strict_entities,
          enable_cot=enable_cot,
        )
        result["layout"] = layout_result
        objects = layout_result.get("objects") if layout_result else None
        if objects:
          object_names = [str(obj.get("name", "")).strip() for obj in objects]
          bboxes = torch.tensor([obj["bbox"] for obj in objects], dtype=torch.float32).unsqueeze(0)
          bboxes = torch.clamp(bboxes, 0.0, 1.0)
      
      # Step 2: 注入 Spatial Adapter（动态按层维度）
      original_processors = None
      try:
        if enable_layout and bboxes is not None and self.load_sd and self.sd_pipe is not None:
          if adapter_container is None:
            adapter_container = create_spatial_adapter_for_kolors()
          elif not isinstance(adapter_container, SpatialAdapterModuleDict):
            container = SpatialAdapterModuleDict()
            if hasattr(adapter_container, "hidden_dim"):
              key = f"dim_{getattr(adapter_container, 'hidden_dim')}"
            else:
              key = "dim_auto"
            container[key] = adapter_container
            adapter_container = container
          
          try:
            unet_device = next(self.sd_pipe.unet.parameters()).device
            bboxes = bboxes.to(unet_device)
          except (StopIteration, AttributeError):
            if hasattr(self, 'sd_device') and self.sd_device != 'cpu':
              unet_device = self.sd_device
              bboxes = bboxes.to(unet_device)
            else:
              unet_device = 'cpu'
              bboxes = bboxes.to(unet_device)
          
          masks = None
          phrase_embeddings = None
          if objects:
            masks = torch.ones((1, len(objects)), device=bboxes.device, dtype=torch.float32)
            if self.is_kolors and not disable_phrase_emb:
              phrase_embeddings = self._encode_objects(object_names or [], unet_device)

          original_processors, spatial_processors, adapter_container = inject_spatial_control_to_unet(
            self.sd_pipe.unet,
            adapter_container,
            bboxes=bboxes,
            phrase_embeddings=phrase_embeddings,
            masks=masks,
            adapter_dtype=torch.float32
          )
          if force_gate and adapter_container is not None:
            forced = 0
            modules = adapter_container.values() if isinstance(adapter_container, SpatialAdapterModuleDict) else [adapter_container]
            for module in modules:
              gated_attn = getattr(module, "gated_attn", None)
              gate = getattr(gated_attn, "gate", None)
              if gate is not None:
                gate.data.fill_(float(gate_value))
                forced += 1
            if forced:
              print(f"   Forced gate={float(gate_value)} on {forced} adapters")
          if adapter_container is not None and hasattr(adapter_container, "set_scale"):
            adapter_container.set_scale(float(adapter_scale))
          print(f"✓ Spatial Adapter 已注入到 {len(spatial_processors)} 个 attention 层")
          print(f"   布局对象数: {len(objects) if objects else 0}")
          print(f"   BBoxes shape: {bboxes.shape}, device: {bboxes.device}")
        
        # Step 3: 语义生成 / 路径选择
        generated_image = None
        if self.is_kolors and self.sd_pipe is not None:
          if use_gill_prompt:
            try:
              semantic_prompt = self._generate_text_prompt(current_prompt, max_new_tokens=max(num_words, 32))
              if semantic_prompt:
                current_prompt = semantic_prompt
              result["semantic_prompt"] = semantic_prompt or current_prompt
              print("🧠 使用 GILL 生成语义 prompt -> Kolors")
            except Exception as e:
              print(f"?? GILL 语义生成失败，回退到原始 prompt: {e}")
              result["semantic_prompt"] = current_prompt
              print("🚀 使用 Kolors 原生文本生成 (Bypass GILLMapper)...")
          else:
            print("🚀 使用 Kolors 原生文本生成 (Bypass GILLMapper)...")
            result["semantic_prompt"] = current_prompt
          original_pad = None
          try:
            # Scheduled Sampling 回调：前半段启用控制，后半段关闭
            callback_kwargs = {}
            def gligen_scheduled_callback(pipe, step_index, timestep, callback_kwargs_in):
              if adapter_container is None:
                return callback_kwargs_in
              if scheduled_sampling_ratio is None:
                return callback_kwargs_in
              total_steps = max(int(num_inference_steps), 1)
              progress = float(step_index) / float(total_steps)
              base_scale = float(adapter_scale)
              target_scale = base_scale if progress < float(scheduled_sampling_ratio) else 0.0
              if hasattr(adapter_container, "set_scale"):
                adapter_container.set_scale(target_scale)
              else:
                for module in adapter_container.modules():
                  if hasattr(module, "set_scale"):
                    module.set_scale(target_scale)
              return callback_kwargs_in

            if scheduled_sampling_ratio is not None:
              callback_kwargs["callback_on_step_end"] = gligen_scheduled_callback

            def _pipe_call(**kwargs):
              try:
                return self.sd_pipe(**kwargs, **callback_kwargs).images[0]
              except TypeError:
                # 兼容不支持 callback_on_step_end 的 diffusers 版本
                return self.sd_pipe(**kwargs).images[0]

            if hasattr(self.sd_pipe, 'tokenizer') and self.sd_pipe.tokenizer is not None and hasattr(self.sd_pipe.tokenizer, '_pad'):
              original_pad = self.sd_pipe.tokenizer._pad
              def compatible_pad(encoded_inputs, max_length=None, padding_strategy=None, pad_to_multiple_of=None, return_attention_mask=None, **kwargs):
                kwargs.pop('padding_side', None)
                return original_pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask)
              self.sd_pipe.tokenizer._pad = compatible_pad
            
            negative_prompt = ""
            try:
              generated_image = _pipe_call(
                prompt=current_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=1024,
                width=1024,
                generator=generator
              )
            except (TypeError, ValueError):
              print("  ℹ️ 尝试不使用 height/width 参数...")
              try:
                generated_image = _pipe_call(
                  prompt=current_prompt,
                  negative_prompt=negative_prompt,
                  guidance_scale=guidance_scale,
                  num_inference_steps=num_inference_steps,
                  generator=generator
                )
              except Exception:
                print("  ℹ️ 尝试不提供 negative_prompt 参数...")
                generated_image = _pipe_call(
                  prompt=current_prompt,
                  guidance_scale=guidance_scale,
                  num_inference_steps=num_inference_steps,
                  generator=generator
                )
          except Exception as e:
            print(f"⚠️ Kolors 直接生成失败: {e}")
            import traceback
            traceback.print_exc()
          finally:
            if original_pad is not None and hasattr(self.sd_pipe, 'tokenizer'):
              self.sd_pipe.tokenizer._pad = original_pad
        else:
          # Non-Kolors fallback: prompt-to-image only.
          if use_gill_prompt:
            try:
              semantic_prompt = self._generate_text_prompt(current_prompt, max_new_tokens=max(num_words, 32))
              if semantic_prompt:
                current_prompt = semantic_prompt
              result["semantic_prompt"] = semantic_prompt or current_prompt
            except Exception as e:
              print(f"?? GILL semantic prompt failed, fallback to original: {e}")
              result["semantic_prompt"] = current_prompt
          else:
            result["semantic_prompt"] = current_prompt

          try:
            generated_image = self.sd_pipe(
              prompt=current_prompt,
              guidance_scale=guidance_scale,
              num_inference_steps=num_inference_steps,
              generator=generator
            ).images[0]
          except Exception as e:
            print(f"?? SD direct generation failed: {e}")
        
        result["image"] = generated_image
        
        if enable_feedback and feedback_verifier is not None and generated_image is not None:
          expected_layout = objects if objects else None
          feedback = feedback_verifier.verify(generated_image, prompt, expected_layout)
          result["feedback"] = feedback
          
          if feedback.get("correct", True):
            result["status"] = "success"
            return result
          
          history.append({"image": generated_image, "feedback": feedback})
          print(f"❌ 验证失败: {feedback.get('feedback')}")
          if attempt == max_retries:
            result["status"] = "failed_max_retries"
            return result
          continue
        else:
          result["status"] = "success"
          return result
      
      finally:
        if enable_layout and original_processors is not None and self.load_sd and self.sd_pipe is not None:
          try:
            remove_spatial_control_from_unet(self.sd_pipe.unet, original_processors)
            print(f"✓ Spatial Adapter 已移除，UNet 已恢复")
          except Exception as e:
            print(f"⚠️ 恢复 Spatial Adapter 时出错: {e}")
    
    if result["status"] is None:
      result["status"] = "failed"
    return result
