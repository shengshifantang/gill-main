from typing import List, Optional
from collections import namedtuple
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
import glob
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from PIL import Image, UnidentifiedImageError
from requests.exceptions import ConnectionError

from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, OPTForCausalLM, AutoModelForCausalLM
from gill import utils
from gill import layers
from gill import layout_planner
from gill import spatial_adapter
from gill import feedback_verifier


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
    vm_was_eval = self.args.freeze_vm and not self.visual_model.training
    
    super(GILLModel, self).train(mode=mode)
    
    if self.args.freeze_lm and lm_was_eval:
      self.lm.eval()
    if self.args.freeze_vm and vm_was_eval:
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

  def generate(self, embeddings = torch.FloatTensor, max_len: int = 32,
               temperature: float = 0.0, top_p: float = 1.0, min_word_tokens: int = 0,
               ret_scale_factor: float = 1.0, gen_scale_factor: float = 1.0,
               filter_value: float = -float('Inf')):
    """Runs greedy decoding and returns generated captions.

    Args:
      min_word_tokens: Minimum number of words to generate before allowing a [IMG] output.
      filter_value: Value to assign to tokens that should never be generated.
    Outputs:
      out: (N, T) int32 sequence of output tokens.
      output_embeddings: (N, T, 256) sequence of text output embeddings.
    """
    self.lm.eval()

    with torch.no_grad():  # no tracking history
      # init output with image tokens
      out = None
      output_embeddings = []
      output_logits = []

      for i in range(max_len):
        output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)

        for idx in self.args.text_emb_layers:
          output_embeddings.append(output.hidden_states[idx])

        logits = output.logits[:, -1, :]  # (N, vocab_size)
        if top_p == 1.0:
          logits = logits.cpu()
        output_logits.append(logits)

        # Prevent the model from generating the [IMG1..n] tokens.
        logits[:, self.retrieval_token_idx[1:]] = filter_value
        logits[:, self.gen_token_idx[1:]] = filter_value

        if (self.retrieval_token_idx or self.gen_token_idx) and self.retrieval_token_idx[0] != -1 and self.gen_token_idx[0] != -1:
          if i < min_word_tokens:
            # Eliminate probability of generating [IMG] if this is earlier than min_word_tokens.
            logits[:, self.retrieval_token_idx] = filter_value
            logits[:, self.gen_token_idx] = filter_value
          else:
            # Multiply by scaling factor.
            if ret_scale_factor > 1:
              logits[:, self.retrieval_token_idx[0]] = logits[:, self.retrieval_token_idx[0]].abs() * ret_scale_factor
            if gen_scale_factor > 1:
              logits[:, self.gen_token_idx[0]] = logits[:, self.gen_token_idx[0]].abs() * gen_scale_factor

        if temperature == 0.0:
          if top_p != 1.0:
            raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
          next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
        else:
          logits = logits / temperature

          # Apply top-p filtering.
          if top_p < 1.0:
            assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # (N, D)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for j in range(sorted_indices.shape[0]):
              indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
              logits[j, indices_to_remove] = filter_value

          token_weights = logits.exp()   # (N, vocab_size)
          next_token = torch.multinomial(token_weights, 1)  # (N, 1)

        # Force generation of the remaining [IMG] tokens if [IMG0] is generated.
        if next_token.shape[0] == 1 and next_token.item() == self.retrieval_token_idx[0]:
          assert self.retrieval_token_idx == self.gen_token_idx, (self.retrieval_token_idx, self.gen_token_idx)
          next_token = torch.tensor(self.retrieval_token_idx)[None, :].long().to(embeddings.device)  # (1, num_tokens)
        else:
          next_token = next_token.long().to(embeddings.device)

        if out is not None:
          out = torch.cat([out, next_token], dim=-1)
        else:
          out = next_token

        next_embedding = self.input_embeddings(next_token)
        embeddings = torch.cat([embeddings, next_embedding], dim=1)

    return out, output_embeddings, output_logits


class GILL(nn.Module):
  def __init__(self, tokenizer, model_args: Optional[GILLArgs] = None,
               path_array: Optional[List[str]] = None, emb_matrix: Optional[torch.tensor] = None,
               load_sd: bool = False, num_gen_images: int = 1, decision_model_path: Optional[str] = None, device_map: Optional[str] = None):
    super().__init__()
    self.model = GILLModel(tokenizer, model_args, device_map=device_map)
    self.path_array = path_array
    self.emb_matrix = emb_matrix
    self.load_sd = load_sd
    self.num_gen_images = num_gen_images
    self.idx2dec = {0: 'gen', 1: 'ret', 2: 'same'}
    self.decision_model = None

    # Load the Stable Diffusion model.
    # 支持指定不同GPU加载图像生成模型
    self.sd_device = os.environ.get('SD_DEVICE', 'cuda:0')  # 默认cuda:0，可通过环境变量指定
    
    # 标记是否使用 Kolors（用于后续生成时的适配）
    self.is_kolors = False
    
    if load_sd:
      # 优先使用Kolors（中文优化），然后是SD v1.5
      kolors_path = "./model/Kolors"
      sd_v15_path = "./model/stable-diffusion-v1-5"
      
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

    if decision_model_path is not None:
      print('Loading decision model...')
      self.decision_model = nn.Sequential(*[
          nn.Dropout(0.5),
          nn.Linear(4096, 2),
      ])
      mlp_checkpoint = torch.load(decision_model_path)
      self.decision_model.load_state_dict(mlp_checkpoint['state_dict'], strict=True)
      self.decision_model.eval()

  def __call__(self, images: Tensor, tgt_tokens: Optional[Tensor] = None, caption_len: Optional[Tensor] = None,
               generate: bool = False, num_words: int = 32, temperature: float = 1.0, top_p: float = 1.0,
               ret_scale_factor: float = 1.0, gen_scale_factor: float = 1.0,
               min_word_tokens: int = 0, mode: str = 'captioning', concat_captions: bool = False,
               input_prefix: Optional[str] = None) -> Tensor:
    if generate:
      return self.model.generate(images, num_words, temperature=temperature, top_p=top_p,
                                 min_word_tokens=min_word_tokens, ret_scale_factor=ret_scale_factor,
                                 gen_scale_factor=gen_scale_factor)
    else:
      output = self.model(
        pixel_values = images,
        labels = tgt_tokens,
        caption_len = caption_len,
        mode = mode,
        concat_captions = concat_captions,
        input_prefix = input_prefix)
      return output

  def generate_for_images_and_texts(
    self, prompts: List, num_words: int = 0, min_word_tokens: int = 0, ret_scale_factor: float = 1.0, gen_scale_factor: float = 1.0,
    top_p: float = 1.0, temperature: float = 0.0, max_num_rets: int = 1, generator=None, 
    always_add_bos : bool = False, guidance_scale: float = 7.5, num_inference_steps: int = 50):
    """
    Encode prompts into embeddings, and generates text and image outputs accordingly.

    Args:
      prompts: List of interleaved PIL.Image.Image and strings representing input to the model.
      num_words: Maximum number of words to generate for. If num_words = 0, the model will run its forward pass and return the outputs.
      min_word_tokens: Minimum number of actual words before generating an image.
      ret_scale_factor: Proportion to scale [IMG] token logits by. A higher value may increase the probability of the model generating [IMG] outputs.
      top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
      temperature: Used to modulate logit distribution.
      max_num_rets: Maximum number of images to return in one generation pass.
    Returns:
      return_outputs: List consisting of either str or List[PIL.Image.Image] objects, representing image-text interleaved model outputs.
    """
    input_embs = []
    input_ids = []
    add_bos = True

    with torch.no_grad():
      for p in prompts:
        if type(p) == Image.Image:
          # Encode as image.
          pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, p)
          pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
          pixel_values = pixel_values[None, ...]

          visual_embs = self.model.get_visual_embs(pixel_values, mode='captioning')  # (1, n_visual_tokens, D)
          input_embs.append(visual_embs)
        elif type(p) == str:
          text_ids = self.model.tokenizer(p, add_special_tokens=add_bos, return_tensors="pt").input_ids.to(self.model.logit_scale.device)
          # Only add <bos> once unless the flag is set.
          if not always_add_bos:
            add_bos = False

          text_embs = self.model.input_embeddings(text_ids)  # (1, T, D)
          input_embs.append(text_embs)
          input_ids.append(text_ids)
        else:
          raise ValueError(f'Input prompts should be either PIL.Image.Image or str types, got {type(p)} instead.')
      input_embs = torch.cat(input_embs, dim=1)
      input_ids = torch.cat(input_ids, dim=1)

      if num_words == 0:
        raise NotImplementedError('Generation not implemented for num_words=0.')
      elif num_words > 0:
        generated_ids, generated_embeddings, _ = self.model.generate(input_embs, num_words, min_word_tokens=min_word_tokens,
          temperature=temperature, top_p=top_p, ret_scale_factor=ret_scale_factor, gen_scale_factor=gen_scale_factor)
        embeddings = generated_embeddings[-1][:, input_embs.shape[1]:]

        # Truncate to newline.
        newline_token_id = self.model.tokenizer('\n', add_special_tokens=False).input_ids[0]
        trunc_idx = 0
        for j in range(generated_ids.shape[1]):
          if generated_ids[0, j] == newline_token_id:
            trunc_idx = j
            break
        if trunc_idx > 0:
          generated_ids = generated_ids[:, :trunc_idx]
          embeddings = embeddings[:, :trunc_idx]
      else:
        raise ValueError

      # Save outputs as an interleaved list.
      return_outputs = []
      # Find up to max_num_rets [IMG] tokens, and their corresponding scores.
      all_ret_idx = [i for i, x in enumerate(generated_ids[0, :] == self.model.retrieval_token_idx[0]) if x][:max_num_rets]
      seen_image_idx = []  # Avoid showing the same image multiple times.

      last_ret_idx = 0
      if len(all_ret_idx) == 0:
        # No [IMG] tokens.
        caption = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return_outputs.append(utils.truncate_caption(caption))
      else:
        for ret_idx in all_ret_idx:
          assert generated_ids[0, ret_idx:ret_idx+self.model.num_tokens].cpu().detach().numpy().tolist() == self.model.retrieval_token_idx, (generated_ids[0, ret_idx:ret_idx+self.model.num_tokens], self.model.retrieval_token_idx)
          raw_emb = embeddings[:, ret_idx:ret_idx+self.model.num_tokens, :]  # (1, 8, 4096)
          assert len(self.model.args.text_emb_layers) == 1

          image_outputs = {
            'gen': [],
            'ret': [],
            'decision': None,
          }

          if self.emb_matrix is not None:
            # Produce retrieval embedding.
            ret_emb = self.model.ret_text_hidden_fcs[0](raw_emb, None)[:, 0, :]  # (1, 256)
            ret_emb = ret_emb / ret_emb.norm(dim=-1, keepdim=True)
            ret_emb = ret_emb.type(self.emb_matrix.dtype)  # (1, 256)
            scores = self.emb_matrix @ ret_emb.T

            # Downweight seen images.
            for seen_idx in seen_image_idx:
              scores[seen_idx, :] -= 1000

            # Get the top 3 images for each image.
            _, top_image_idx = scores.squeeze().topk(3)
            for img_idx in top_image_idx:
              # Find the first image that does not error out.
              try:
                seen_image_idx.append(img_idx)
                img = utils.get_image_from_url(self.path_array[img_idx])
                image_outputs['ret'].append((img, 'ret', scores[img_idx].item()))
                if len(image_outputs) == max_num_rets:
                  break
              except (UnidentifiedImageError, ConnectionError, OSError):
                pass

            # Make decision with MLP.
            if self.decision_model is not None:
              decision_emb = raw_emb[:, 0, :]  # (1, 4096)
              assert decision_emb.shape[1] == 4096, decision_emb.shape
              decision_logits = self.decision_model(decision_emb)
              probs = decision_logits.softmax(dim=-1).cpu().float().numpy().tolist()
              image_outputs['decision'] = [self.idx2dec[decision_logits.argmax().item()]] + probs
          else:
            # If no embedding matrix is provided, generate instead.
            image_outputs['decision'] = ['gen', [0, 1]]

          # Produce generation embedding.
          gen_prefix = ''.join([f'[IMG{i}]' for i in range(self.model.args.num_tokens)])
          gen_prefx_ids = self.model.tokenizer(gen_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.logit_scale.device)
          gen_prefix_embs = self.model.input_embeddings(gen_prefx_ids)  # (1, T, D)
          
          # 检查是否需要 pooled embedding (Kolors 配置)
          gen_fc = self.model.gen_text_hidden_fcs[0]
          pooled_emb = None
          if hasattr(gen_fc, 'output_pooled') and gen_fc.output_pooled:
            gen_emb, pooled_emb = gen_fc(raw_emb, gen_prefix_embs, return_pooled=True)
            # gen_emb: (1, num_clip_tokens, gen_emb_dim)
            # pooled_emb: (1, gen_emb_dim)
          else:
            gen_emb = gen_fc(raw_emb, gen_prefix_embs)  # (1, 77, 768)
          
          # 获取目标维度
          target_seq_len = self.model.args.num_clip_tokens
          target_emb_dim = self.model.args.gen_emb_dim

          if gen_emb.shape[1] != target_seq_len:
            print(f"Padding {gen_emb.shape} to target ({target_seq_len}, {target_emb_dim})")
            bs = gen_emb.shape[0]
            gen_emb = gen_emb.reshape(bs, -1, target_emb_dim)  # (bs, T, D)
            seq_len = gen_emb.shape[1]
            if seq_len < target_seq_len:
              gen_emb = torch.cat([gen_emb, torch.zeros((bs, target_seq_len - seq_len, target_emb_dim), device=gen_emb.device, dtype=gen_emb.dtype)], dim=1)
            print('Padded to', gen_emb.shape)

          gen_emb = gen_emb.repeat(self.num_gen_images, 1, 1)  # (self.num_gen_images, seq_len, emb_dim)
          if pooled_emb is not None:
            pooled_emb = pooled_emb.repeat(self.num_gen_images, 1)  # (self.num_gen_images, emb_dim)

          # OPTIM(jykoh): Only generate if scores are low.
          if self.load_sd:
            # If num_gen_images > 8, split into multiple batches (for GPU memory reasons).
            gen_max_bs = 8
            gen_images = []
            
            # 获取生成前的文本作为 prompt（备用）
            text_prompt = self.model.tokenizer.batch_decode(
                generated_ids[:, last_ret_idx:ret_idx], 
                skip_special_tokens=True
            )[0]
            
            for i in range(0, self.num_gen_images, gen_max_bs):
              if self.is_kolors:
                # Kolors 模式
                if pooled_emb is not None and gen_emb.shape[-1] == 2048:
                  # 使用训练好的 embedding (Kolors 适配版)
                  print(f"使用 Kolors embedding 生成: prompt_embeds={gen_emb.shape}, pooled={pooled_emb.shape}")
                  gen_images.extend(
                    self.sd_pipe(
                      prompt_embeds=gen_emb[i:i+gen_max_bs].half(),
                      pooled_prompt_embeds=pooled_emb[i:i+gen_max_bs].half(),
                      generator=generator,
                      guidance_scale=guidance_scale, 
                      num_inference_steps=num_inference_steps,
                      height=1024,
                      width=1024
                    ).images)
                else:
                  # 回退到文本 prompt（旧模型或维度不匹配）
                  print(f"使用文本 prompt 生成: {text_prompt[:50]}...")
                  gen_images.extend(
                    self.sd_pipe(
                      prompt=text_prompt if text_prompt else "一张美丽的图片",
                      generator=generator,
                      guidance_scale=guidance_scale, 
                      num_inference_steps=num_inference_steps,
                      height=1024,
                      width=1024
                    ).images)
              else:
                # SD v1.5 使用 prompt_embeds
                gen_images.extend(
                  self.sd_pipe(prompt_embeds=gen_emb[i:i+gen_max_bs], generator=generator,
                               guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images)

            all_gen_pixels = []
            for img in gen_images:
              pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, img.resize((224, 224)).convert('RGB'))
              pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
              all_gen_pixels.append(pixel_values)
            
            if self.emb_matrix is not None:
              all_gen_pixels = torch.stack(all_gen_pixels, dim=0)
              gen_visual_embs = self.model.get_visual_embs(all_gen_pixels, mode='retrieval')  # (1, D)
              gen_visual_embs = gen_visual_embs / gen_visual_embs.norm(dim=-1, keepdim=True)
              gen_visual_embs = gen_visual_embs.type(self.emb_matrix.dtype)
              gen_rank_scores = (gen_visual_embs @ ret_emb.T).squeeze()
              sorted_score_idx = torch.argsort(-gen_rank_scores)

              # Rank images by retrieval score.
              if self.num_gen_images > 1:
                image_outputs['gen'] = [(gen_images[idx], gen_rank_scores[idx].item()) for idx in sorted_score_idx]
              else:
                image_outputs['gen'] = [(gen_images[0], gen_rank_scores.item())]
            else:
              image_outputs['gen'] = [(gen_images[0], 0)]
          else:
            image_outputs['gen'] = [gen_emb]

          caption = self.model.tokenizer.batch_decode(generated_ids[:, last_ret_idx:ret_idx], skip_special_tokens=True)[0]
          last_ret_idx = ret_idx + 1
          return_outputs.append(utils.truncate_caption(caption) + f' {gen_prefix}')
          return_outputs.append(image_outputs)

    return return_outputs

  def get_log_likelihood_scores(
    self, prompts: List):
    """
    Output the log likelihood of the given interleaved prompts.

    Args:
      prompts: List of interleaved PIL.Image.Image and strings representing input to the model.
    Returns:
      Log likelihood score of the prompt sequence.
    """
    input_embs = []
    input_ids = []
    add_bos = True

    for p in prompts:
      if type(p) == Image.Image:
        # Encode as image.
        pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, p)
        pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
        pixel_values = pixel_values[None, ...]

        visual_embs = self.model.get_visual_embs(pixel_values, mode='captioning')  # (1, n_visual_tokens, D)
        input_embs.append(visual_embs)
        id_ = torch.zeros(visual_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100
        input_ids.append(id_)
      elif type(p) == str:
        text_ids = self.model.tokenizer(p, add_special_tokens=True, return_tensors="pt").input_ids.to(self.model.logit_scale.device)
        if not add_bos:
          # Remove <bos> tag.
          text_ids = text_ids[:, 1:]
        else:
          # Only add <bos> once.
          add_bos = False

        text_embs = self.model.input_embeddings(text_ids)  # (1, T, D)
        input_embs.append(text_embs)
        input_ids.append(text_ids)
      else:
        raise ValueError(f'Input prompts should be either PIL.Image.Image or str types, got {type(p)} instead.')
    input_embs = torch.cat(input_embs, dim=1)
    input_ids = torch.cat(input_ids, dim=1)

    outputs = self.model.lm(inputs_embeds=input_embs, labels=input_ids, use_cache=False, output_hidden_states=True)
    return -outputs.loss.item()  

  def generate_with_layout(self, prompt: str, enable_layout: bool = True, 
                          enable_feedback: bool = False, layout_planner=None,
                          spatial_adapter=None, feedback_verifier=None,
                          num_words: int = 15, guidance_scale: float = 7.5,
                          num_inference_steps: int = 50, generator=None,
                          max_retries: int = 3):
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
      layout_result = None
      if enable_layout and layout_planner is not None:
        layout_result = layout_planner.generate_layout(current_prompt, apply_refinement=True)
        result["layout"] = layout_result
        objects = layout_result.get("objects") if layout_result else None
        if objects:
          bboxes = torch.tensor([obj["bbox"] for obj in objects], dtype=torch.float32).unsqueeze(0)
      
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
              bboxes = bboxes.to(self.sd_device)
            else:
              bboxes = bboxes.to('cpu')
          
          original_processors, spatial_processors, adapter_container = inject_spatial_control_to_unet(
            self.sd_pipe.unet,
            adapter_container,
            bboxes=bboxes,
            phrase_embeddings=None
          )
          print(f"✓ Spatial Adapter 已注入到 {len(spatial_processors)} 个 attention 层")
          print(f"   布局对象数: {len(objects) if objects else 0}")
          print(f"   BBoxes shape: {bboxes.shape}, device: {bboxes.device}")
        
        # Step 3: 语义生成 / 路径选择
        generated_image = None
        if self.is_kolors and self.sd_pipe is not None:
          print("🚀 使用 Kolors 原生文本生成 (Bypass GILLMapper)...")
          original_pad = None
          try:
            if hasattr(self.sd_pipe, 'tokenizer') and self.sd_pipe.tokenizer is not None and hasattr(self.sd_pipe.tokenizer, '_pad'):
              original_pad = self.sd_pipe.tokenizer._pad
              def compatible_pad(encoded_inputs, max_length=None, padding_strategy=None, pad_to_multiple_of=None, return_attention_mask=None, **kwargs):
                kwargs.pop('padding_side', None)
                return original_pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask)
              self.sd_pipe.tokenizer._pad = compatible_pad
            
            negative_prompt = ""
            try:
              generated_image = self.sd_pipe(
                prompt=current_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=1024,
                width=1024,
                generator=generator
              ).images[0]
            except (TypeError, ValueError):
              print("  ℹ️ 尝试不使用 height/width 参数...")
              try:
                generated_image = self.sd_pipe(
                  prompt=current_prompt,
                  negative_prompt=negative_prompt,
                  guidance_scale=guidance_scale,
                  num_inference_steps=num_inference_steps,
                  generator=generator
                ).images[0]
              except Exception:
                print("  ℹ️ 尝试不提供 negative_prompt 参数...")
                generated_image = self.sd_pipe(
                  prompt=current_prompt,
                  guidance_scale=guidance_scale,
                  num_inference_steps=num_inference_steps,
                  generator=generator
                ).images[0]
          except Exception as e:
            print(f"⚠️ Kolors 直接生成失败: {e}")
            import traceback
            traceback.print_exc()
          finally:
            if original_pad is not None and hasattr(self.sd_pipe, 'tokenizer'):
              self.sd_pipe.tokenizer._pad = original_pad
          result["semantic_prompt"] = current_prompt
        else:
          model_outputs = self.generate_for_images_and_texts(
            [current_prompt],
            num_words=num_words,
            ret_scale_factor=1.0,
            gen_scale_factor=1.0,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
          )
          
          for output in model_outputs:
            if isinstance(output, dict) and 'gen' in output and 'text' in output:
              result["semantic_prompt"] = output['text']
              break
          
          for output in model_outputs:
            if isinstance(output, dict) and 'gen' in output and len(output['gen']) > 0:
              generated_image = output['gen'][0][0]
              break
        
        result["image"] = generated_image
        
        if enable_feedback and feedback_verifier is not None and generated_image is not None:
          expected_layout = {"objects": objects} if objects else None
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


def load_gill(model_dir: str, load_ret_embs: bool = True, decision_model_fn: str = 'decision_model.pth.tar', load_sd: bool = True, device_map: Optional[str] = None) -> GILL:
  model_args_path = os.path.join(model_dir, 'model_args.json')
  model_ckpt_path = os.path.join(model_dir, 'pretrained_ckpt.pth.tar')
  embs_paths = [s for s in glob.glob(os.path.join(model_dir, 'cc3m*.npy'))]

  if not os.path.exists(model_args_path):
    raise ValueError(f'model_args.json does not exist in {model_dir}.')
  if not os.path.exists(model_ckpt_path):
    raise ValueError(f'pretrained_ckpt.pth.tar does not exist in {model_dir}.')
  if not load_ret_embs or len(embs_paths) == 0:
    if len(embs_paths) == 0:
      print(f'cc3m.npy files do not exist in {model_dir}.')
    print('Running the model without retrieval.')
    path_array, emb_matrix = None, None
  else:
    # Load embeddings.
    # Construct embedding matrix for nearest neighbor lookup.
    path_array = []
    emb_matrix = []

    # These were precomputed for all CC3M images with `model.get_visual_embs(image, mode='retrieval')`.
    for p in embs_paths:
      with open(p, 'rb') as wf:
          train_embs_data = pkl.load(wf)
          path_array.extend(train_embs_data['paths'])
          emb_matrix.extend(train_embs_data['embeddings'])
    emb_matrix = np.stack(emb_matrix, axis=0)

    # Number of paths should be equal to number of embeddings.
    assert len(path_array) == emb_matrix.shape[0], (len(path_array), emb_matrix.shape)

  with open(model_args_path, 'r') as f:
    model_kwargs = json.load(f)

  # 修复 opt_version 路径（如果路径不存在，尝试自动检测），默认使用 Qwen2.5-7B-Instruct
  opt_version = model_kwargs.get('opt_version', 'Qwen/Qwen2.5-7B-Instruct')
  
  if opt_version.startswith('./') or opt_version.startswith('../'):
    # 相对路径，检查是否存在（需要转换为绝对路径）
    abs_opt_version = os.path.abspath(opt_version) if not os.path.isabs(opt_version) else opt_version
    
    if not os.path.exists(abs_opt_version):
      # 尝试常见的替代路径
      base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
      alternative_paths = [
        os.path.join(base_dir, 'model', 'Qwen2.5-7B-Instruct'),
        os.path.join(base_dir, 'model', 'qwen2.5-7b-instruct'),
        os.path.join(base_dir, 'model', 'Qwen', 'Qwen2.5-7B-Instruct'),
        './model/Qwen2.5-7B-Instruct',
        './model/qwen2.5-7b-instruct',
        './model/Qwen/Qwen2.5-7B-Instruct',
        'model/Qwen2.5-7B-Instruct',
        'model/qwen2.5-7b-instruct',
        'model/Qwen/Qwen2.5-7B-Instruct'
      ]
      found = False
      
      for alt_path in alternative_paths:
        abs_alt_path = os.path.abspath(alt_path) if not os.path.isabs(alt_path) else alt_path
        if os.path.exists(abs_alt_path):
          print(f"⚠️  原始路径不存在: {opt_version}")
          print(f"   自动切换到: {alt_path}")
          # 使用绝对路径，确保后续检查正确
          opt_version = abs_alt_path
          model_kwargs['opt_version'] = opt_version
          found = True
          break
      if not found:
        print(f"⚠️  警告: opt_version 路径不存在: {opt_version}")
        print(f"   尝试使用 HuggingFace Hub 路径...")
        # 如果都不存在，默认使用官方 Qwen Hub 路径
        opt_version = 'Qwen/Qwen2.5-7B-Instruct'
        model_kwargs['opt_version'] = opt_version
        print(f"   使用: {opt_version}")

  # 修复 visual_encoder 路径（如果路径不存在，尝试自动检测）
  visual_encoder = model_kwargs.get('visual_encoder', 'openai/clip-vit-large-patch14')
  
  if visual_encoder.startswith('./') or visual_encoder.startswith('../'):
    # 相对路径，检查是否存在（需要转换为绝对路径）
    abs_visual_encoder = os.path.abspath(visual_encoder) if not os.path.isabs(visual_encoder) else visual_encoder
    
    if not os.path.exists(abs_visual_encoder):
      # 尝试常见的替代路径
      base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
      alternative_paths = [
        os.path.join(base_dir, 'model', 'chinese_clip_ViT-L-14'),
        os.path.join(base_dir, 'model', 'chinese-clip-vit-large-patch14'),
        './model/chinese_clip_ViT-L-14',
        './model/chinese-clip-vit-large-patch14',
        'model/chinese_clip_ViT-L-14',
        'model/chinese-clip-vit-large-patch14'
      ]
      found = False
      
      for alt_path in alternative_paths:
        abs_alt_path = os.path.abspath(alt_path) if not os.path.isabs(alt_path) else alt_path
        if os.path.exists(abs_alt_path):
          print(f"⚠️  原始 visual_encoder 路径不存在: {visual_encoder}")
          print(f"   自动切换到: {alt_path}")
          # 使用绝对路径，确保后续检查正确
          visual_encoder = abs_alt_path
          model_kwargs['visual_encoder'] = visual_encoder
          found = True
          break
      if not found:
        print(f"⚠️  警告: visual_encoder 路径不存在: {visual_encoder}")
        print(f"   尝试使用 HuggingFace Hub 路径...")
        # 如果都不存在，尝试使用 HuggingFace Hub
        if 'chinese' in visual_encoder.lower() or 'clip' in visual_encoder.lower():
          visual_encoder = 'OFA-Sys/chinese-clip-vit-large-patch14'
        else:
          visual_encoder = 'openai/clip-vit-large-patch14'  # 默认
        model_kwargs['visual_encoder'] = visual_encoder
        print(f"   使用: {visual_encoder}")

  # Initialize tokenizer.
  # 检查是否是本地路径（相对路径或绝对路径）
  _is_local = False
  if opt_version.startswith('./') or opt_version.startswith('../') or os.path.isabs(opt_version):
    _abs_path = os.path.abspath(opt_version) if not os.path.isabs(opt_version) else opt_version
    _is_local = os.path.exists(_abs_path) and os.path.isdir(_abs_path)
  # Hub 路径（如 'deepseek-ai/deepseek-llm-7b-base'）不是本地路径
  
  tokenizer = AutoTokenizer.from_pretrained(opt_version, use_fast=False, local_files_only=_is_local, trust_remote_code=True)
  if tokenizer.pad_token is None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
  # Add an image token for loss masking (and visualization) purposes.
  tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer

  # Add [IMG] tokens to the vocabulary.
  model_kwargs['retrieval_token_idx'] = []
  for i in range(model_kwargs['num_tokens']):
      print(f'Adding [IMG{i}] token to vocabulary.')
      print(f'Before adding new token, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
      num_added_tokens = tokenizer.add_tokens(f'[IMG{i}]')
      print(f'After adding {num_added_tokens} new tokens, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
      ret_token_idx = tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
      assert len(ret_token_idx) == 1, ret_token_idx
      model_kwargs['retrieval_token_idx'].append(ret_token_idx[0])
  # Use the same RET tokens for generation.
  model_kwargs['gen_token_idx'] = model_kwargs['retrieval_token_idx']

  args = namedtuple('args', model_kwargs)(**model_kwargs)

  # Load decision model.
  if decision_model_fn is not None:
    decision_model_path = os.path.join(model_dir, decision_model_fn)
  else:
    decision_model_path = None

  # Initialize model for inference.
  model = GILL(tokenizer, args, path_array=path_array, emb_matrix=emb_matrix,
               load_sd=load_sd, num_gen_images=1, decision_model_path=decision_model_path, device_map=device_map)
  model = model.eval()
  
  # Load pretrained linear mappings and [IMG] embeddings (在移到GPU之前加载)
  print("📦 加载 checkpoint 到 CPU...")
  checkpoint = torch.load(model_ckpt_path, map_location='cpu')
  state_dict = {}
  # This is needed if we train with DDP.
  for k, v in checkpoint['state_dict'].items():
      state_dict[k.replace('module.', '')] = v
  img_token_embeddings = state_dict['model.input_embeddings.weight'].cpu().detach()
  del state_dict['model.input_embeddings.weight']

  print("📥 加载模型权重...")
  model.load_state_dict(state_dict, strict=False)
  
  # 处理多GPU情况：如果使用device_map="auto"，模型已经在多GPU上了
  if device_map is not None and isinstance(device_map, str) and device_map.startswith("cuda") and "," in device_map:
    # 多GPU模式，模型已经通过device_map分布到多GPU
    print("🚀 模型已通过 device_map 分布到多GPU")
    model = model.bfloat16()
  else:
    # 单GPU模式，正常移到GPU
    print("🚀 将模型移到 GPU...")
    model = model.bfloat16()
    model = model.cuda()
  
  # Copy over the embeddings of the [IMG] tokens (while loading the others from the pretrained LLM).
  print("🎨 设置 IMG token embeddings...")
  with torch.no_grad():
      if 'share_ret_gen' in model_kwargs:
        assert model_kwargs['share_ret_gen'], 'Model loading only supports share_ret_gen=True for now.'
      # img_token_embeddings包含所有token embeddings，只取最后num_tokens个
      img_tokens_to_copy = img_token_embeddings[-model_kwargs['num_tokens']:, :].to(model.model.input_embeddings.weight.device)
      model.model.input_embeddings.weight[-model_kwargs['num_tokens']:, :].copy_(img_tokens_to_copy)

  if load_ret_embs and len(embs_paths) > 0:
    logit_scale = model.model.logit_scale.exp()
    emb_matrix = torch.tensor(emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
    emb_matrix = emb_matrix / emb_matrix.norm(dim=1, keepdim=True)
    emb_matrix = logit_scale * emb_matrix
    model.emb_matrix = emb_matrix

  return model
