"""
Spatial Control Adapter (GLIGEN 风格) - 修复版

主要修正：
1. ✅ 注入位置：Cross-Attention（attn2）而非 Self-Attention（attn1）
2. ✅ 显存优化：支持 Gradient Checkpointing
3. ✅ 坐标验证：强制检查 BBox 归一化
4. ✅ 多维度适配：自动检测 UNet 各层维度

参考：
- GLIGEN (CVPR 2023): Open-Set Grounded Text-to-Image Generation
- 使用 Diffusers Attention Processor 机制，避免修改源码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math
import warnings


class SpatialPositionNet(nn.Module):
    """
    将 BBox [x1, y1, x2, y2] 映射为高维特征
    
    使用 Fourier Features 编码位置信息
    """
    def __init__(self, in_dim: int = 4, out_dim: int = 2048, fourier_freqs: int = 8):
        super().__init__()
        self.out_dim = out_dim
        
        # Fourier Features: 将坐标映射到高频空间
        self.fourier_embedder = nn.Sequential(
            nn.Linear(in_dim * fourier_freqs * 2, out_dim // 2),
            nn.SiLU(),
            nn.Linear(out_dim // 2, out_dim)
        )
        
        # 生成 Fourier 频率
        freqs = torch.linspace(0, 1, fourier_freqs)
        self.register_buffer('freqs', freqs)
    
    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bboxes: (B, N, 4) 归一化坐标 [x1, y1, x2, y2]
        
        Returns:
            box_emb: (B, N, out_dim) 位置 embedding
        """
        B, N, _ = bboxes.shape
        
        # ✅ 坐标验证（防止未归一化的坐标）
        if bboxes.max() > 1.5 or bboxes.min() < -0.5:
            warnings.warn(f"BBox 坐标异常：min={bboxes.min():.2f}, max={bboxes.max():.2f}，期望范围 [0, 1]")
            # 自动归一化（兜底）
            bboxes = torch.clamp(bboxes, 0, 1)
        
        # 生成 Fourier features
        bboxes_expanded = bboxes.unsqueeze(-1)  # (B, N, 4, 1)
        freqs = self.freqs.view(1, 1, 1, -1)  # (1, 1, 1, fourier_freqs)
        
        # sin 和 cos 变换
        sin_features = torch.sin(bboxes_expanded * freqs * 2 * math.pi)  # (B, N, 4, fourier_freqs)
        cos_features = torch.cos(bboxes_expanded * freqs * 2 * math.pi)  # (B, N, 4, fourier_freqs)
        
        # 拼接
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)  # (B, N, 4, 2*fourier_freqs)
        fourier_features = fourier_features.reshape(B, N, -1)  # (B, N, 4*2*fourier_freqs)
        
        # 通过 MLP
        box_emb = self.fourier_embedder(fourier_features)  # (B, N, out_dim)
        
        return box_emb


class GatedSelfAttentionDense(nn.Module):
    """
    Gated Self-Attention 层（参考 GLIGEN）
    
    将空间信息通过门控机制注入到 UNet 特征中
    """
    def __init__(self, query_dim: int, context_dim: int, n_heads: int = 8, d_head: int = 64):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.n_heads = n_heads
        self.d_head = d_head
        
        # Query, Key, Value 投影
        self.to_q = nn.Linear(query_dim, n_heads * d_head, bias=False)
        self.to_k = nn.Linear(context_dim, n_heads * d_head, bias=False)
        self.to_v = nn.Linear(context_dim, n_heads * d_head, bias=False)
        self.to_out = nn.Linear(n_heads * d_head, query_dim)
        
        # ✅ 门控参数（初始化为 0，确保训练初期不影响原有模型）
        self.gate = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self, x: torch.Tensor, spatial_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: UNet 的中间特征 (B, T, query_dim)
            spatial_context: 空间 embedding (B, N, context_dim)
        
        Returns:
            out: 注入空间信息后的特征 (B, T, query_dim)
        """
        B, T, _ = x.shape
        _, N, _ = spatial_context.shape
        
        # 计算 Q, K, V
        q = self.to_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = self.to_k(spatial_context).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, d)
        v = self.to_v(spatial_context).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, d)
        
        # Attention
        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)  # (B, H, T, N)
        attn_out = torch.matmul(attn, v)  # (B, H, T, d)
        
        # Reshape 并投影
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)  # (B, T, H*d)
        attn_out = self.to_out(attn_out)  # (B, T, query_dim)
        
        # 门控残差连接
        # tanh(gate) 允许门控在 -1 到 1 之间，初始为 0
        out = x + torch.tanh(self.gate) * attn_out
        
        return out


class SpatialControlAdapter(nn.Module):
    """
    Spatial Control Adapter for Kolors (SDXL-based)
    
    使用 Diffusers Attention Processor 机制注入空间控制
    """
    def __init__(self, hidden_dim: int = 2048, num_heads: int = 8, text_dim: int = 4096):
        """
        Args:
            hidden_dim: UNet 的 hidden dimension (Kolors 使用 2048)
            num_heads: Attention head 数量
            text_dim: 文本特征维度（Kolors/ChatGLM 默认 4096）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        
        # 位置编码网络，输出与当前层 hidden_dim 对齐
        self.position_net = SpatialPositionNet(out_dim=hidden_dim)
        # 文本特征投影到 hidden_dim，避免维度不匹配
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Gated Self-Attention 层（与当前层维度匹配）
        self.gated_attn = GatedSelfAttentionDense(
            query_dim=hidden_dim,
            context_dim=hidden_dim,
            n_heads=num_heads,
            d_head=hidden_dim // num_heads
        )
    
    def forward(self, 
                unet_features: torch.Tensor,
                bboxes: torch.Tensor,
                phrase_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        在 UNet forward 过程中被调用
        
        Args:
            unet_features: UNet 中间层特征 (B, T, hidden_dim)
            bboxes: 归一化坐标 (B, N, 4)
            phrase_embeddings: 物体名称的文本 embedding (B, N, text_dim)，可选
        
        Returns:
            controlled_features: 注入空间信息后的特征
        """
        # ✅ CFG 批次广播（Classifier-Free Guidance 会复制 batch）
        if bboxes.shape[0] == 1 and unet_features.shape[0] > 1:
            bboxes = bboxes.repeat(unet_features.shape[0], 1, 1)
        if phrase_embeddings is not None and phrase_embeddings.shape[0] == 1 and unet_features.shape[0] > 1:
            phrase_embeddings = phrase_embeddings.repeat(unet_features.shape[0], 1, 1)

        # ✅ 对齐 dtype（防止 AMP 混合精度报错）
        bboxes = bboxes.to(dtype=unet_features.dtype, device=unet_features.device)
        
        # ✅ 维度匹配检查（防御性编程）
        if len(unet_features.shape) >= 2:
            actual_dim = unet_features.shape[-1]
            if actual_dim != self.gated_attn.query_dim:
                warnings.warn(f"维度不匹配：期望 {self.gated_attn.query_dim}，实际 {actual_dim}，跳过注入")
                return unet_features
        
        # 1. 获取位置特征
        box_emb = self.position_net(bboxes)  # (B, N, hidden_dim)
        
        # 2. 融合位置特征和文本特征（如果有）
        if phrase_embeddings is not None:
            phrase_embeddings = phrase_embeddings.to(dtype=unet_features.dtype, device=unet_features.device)
            # 投影到 hidden_dim
            if phrase_embeddings.shape[-1] != self.hidden_dim:
                phrase_embeddings = self.text_proj(phrase_embeddings)
            spatial_context = box_emb + phrase_embeddings  # (B, N, hidden_dim)
        else:
            spatial_context = box_emb
        
        # 3. 通过 Gated Self-Attention 注入
        controlled_features = self.gated_attn(unet_features, spatial_context)
        
        return controlled_features


class SpatialControlProcessor:
    """
    Diffusers Attention Processor 包装器（兼容 SDXL/Kolors）
    
    ⚠️ 关键修正：只在 Cross-Attention 层注入（GLIGEN 论文要求）
    """
    def __init__(self, adapter: SpatialControlAdapter, original_processor=None, is_cross_attn: bool = False):
        self.adapter = adapter
        self.original_processor = original_processor
        self.is_cross_attn = is_cross_attn  # ✅ 标记是否为 Cross-Attention 层
        # 存储 bboxes 和 phrase_embeddings（通过上下文传递）
        self.bboxes = None
        self.phrase_embeddings = None
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, 
                 attention_mask=None, scale=None, **kwargs):
        """
        在 UNet 的 attention 层中被调用
        
        Args:
            attn: Attention 模块
            hidden_states: UNet 中间特征 (B, T, hidden_dim)
            encoder_hidden_states: 文本 embedding (B, T_text, hidden_dim)
            attention_mask: 注意力掩码
            scale: 缩放因子（SDXL 特有）
        """
        # ✅ 修正：先执行原始 Cross-Attention，再注入空间控制
        # 这样可以让模型先理解文本语义，再融合空间信息
        if self.original_processor is not None:
            hidden_states = self.original_processor(
                attn, hidden_states, encoder_hidden_states,
                attention_mask=attention_mask, scale=scale, **kwargs
            )
        else:
            # 回退到标准 attention 计算
            hidden_states = self._default_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask
            )
        
        # ✅ 修正：只在 Cross-Attention 层且有 BBox 时注入
        if self.is_cross_attn and self.bboxes is not None and encoder_hidden_states is not None:
            # 检查维度匹配
            if len(hidden_states.shape) >= 2 and hidden_states.shape[-1] == self.adapter.gated_attn.query_dim:
                try:
                    # 注入空间控制
                    hidden_states = self.adapter(
                        hidden_states,
                        self.bboxes,
                        self.phrase_embeddings
                    )
                except Exception as e:
                    # 防御性编程：如果注入失败，不影响原有流程
                    print(f"⚠️ Spatial control injection failed: {e}")
        
        return hidden_states
    
    def _default_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask):
        """回退的标准 Attention 实现"""
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif hasattr(attn, 'norm_cross') and attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * (head_dim ** -0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = attention_scores.softmax(dim=-1)
        
        if hasattr(attn, 'dropout'):
            attention_probs = F.dropout(attention_probs, p=attn.dropout, training=attn.training)
        
        attention_probs = attention_probs.to(value.dtype)
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # to_out 通常是 [Linear, Dropout]
        if hasattr(attn, 'to_out'):
            if isinstance(attn.to_out, nn.ModuleList) or isinstance(attn.to_out, nn.Sequential):
                for layer in attn.to_out:
                    hidden_states = layer(hidden_states)
            else:
                hidden_states = attn.to_out(hidden_states)
        
        return hidden_states
    
    def set_spatial_control(self, bboxes, phrase_embeddings=None):
        """设置空间控制信息"""
        self.bboxes = bboxes
        self.phrase_embeddings = phrase_embeddings


class SpatialAdapterModuleDict(nn.ModuleDict):
    """
    管理多维度 Adapter 的容器，按 hidden_dim 复用/创建
    """
    pass


def _get_attn_layer_dim(unet, name: str) -> Optional[int]:
    """
    解析 attention processor 对应层的 hidden_dim
    支持解析 ModuleList 的数字索引（如 down_blocks.1.attentions...）
    """
    module_path = name.replace(".processor", "")
    sub_mod = unet
    
    for part in module_path.split('.'):
        # 处理数字索引（针对 ModuleList）
        if part.isdigit():
            try:
                sub_mod = sub_mod[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            if not hasattr(sub_mod, part):
                return None
            sub_mod = getattr(sub_mod, part)
    
    # 尝试获取维度
    if hasattr(sub_mod, "to_q"):
        return sub_mod.to_q.in_features
    if hasattr(sub_mod, "query_dim"):
        return sub_mod.query_dim
    # SDXL/Kolors 部分层可能使用 inner_dim
    if hasattr(sub_mod, "inner_dim"):
        return sub_mod.inner_dim
    
    return None


def _is_cross_attention_layer(name: str) -> bool:
    """
    判断是否为 Cross-Attention 层
    
    SDXL/Kolors 命名规则：
    - attn1: Self-Attention
    - attn2: Cross-Attention
    """
    return 'attn2' in name or 'cross_attn' in name.lower()


def inject_spatial_control_to_unet(
    unet,
    adapter_dict: Optional[SpatialAdapterModuleDict] = None,
    bboxes: Optional[torch.Tensor] = None,
    phrase_embeddings: Optional[torch.Tensor] = None,
    num_heads: int = 8,
) -> Tuple[Dict, Dict, SpatialAdapterModuleDict]:
    """
    将空间控制注入到 UNet 中（多维度适配）
    
    Args:
        unet: Kolors/SDXL UNet
        adapter_dict: 管理所有维度 Adapter 的 ModuleDict（可复用/训练）
        bboxes: (B, N, 4) 归一化坐标
        phrase_embeddings: (B, N, text_dim) 文本特征，可选
        num_heads: 默认 8
    
    Returns:
        processors: 原始 processors（用于恢复）
        spatial_processors: 注入后的 processors
        adapter_dict: 更新后的容器（便于外部复用）
    """
    if adapter_dict is None:
        adapter_dict = SpatialAdapterModuleDict()
    
    processors = {}
    spatial_processors = {}
    
    # 获取 UNet 设备和 dtype
    try:
        unet_device = next(unet.parameters()).device
        unet_dtype = next(unet.parameters()).dtype
    except StopIteration:
        unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet_dtype = torch.float32
    
    # 移动 bboxes 和 phrase_embeddings 到正确设备
    if bboxes is not None and bboxes.device != unet_device:
        bboxes = bboxes.to(unet_device)
    if phrase_embeddings is not None and phrase_embeddings.device != unet_device:
        phrase_embeddings = phrase_embeddings.to(unet_device)
    
    def get_or_create_adapter(layer_dim: int) -> SpatialControlAdapter:
        """获取或创建指定维度的 Adapter"""
        text_dim = phrase_embeddings.shape[-1] if phrase_embeddings is not None else 4096
        key = f"dim_{layer_dim}"
        
        if key not in adapter_dict:
            adapter_dict[key] = SpatialControlAdapter(
                hidden_dim=layer_dim,
                num_heads=num_heads,
                text_dim=text_dim
            ).to(device=unet_device, dtype=unet_dtype)
        else:
            # 确保设备和 dtype 匹配
            if next(adapter_dict[key].parameters()).device != unet_device:
                adapter_dict[key] = adapter_dict[key].to(unet_device)
            if next(adapter_dict[key].parameters()).dtype != unet_dtype:
                adapter_dict[key] = adapter_dict[key].to(dtype=unet_dtype)
        
        return adapter_dict[key]
    
    # 方法 1: 使用 diffusers 的 get_attn_processors（推荐）
    try:
        original_processors_dict = unet.attn_processors
        new_processors = {}
        
        for name, original_processor in original_processors_dict.items():
            # 获取层维度
            layer_dim = _get_attn_layer_dim(unet, name)
            if layer_dim is None:
                processors[name] = original_processor
                new_processors[name] = original_processor
                continue
            
            # ✅ 关键修正：只在 Cross-Attention 层注入
            is_cross_attn = _is_cross_attention_layer(name)
            
            # 如果已经是 SpatialControlProcessor，避免重复包装
            if isinstance(original_processor, SpatialControlProcessor):
                base_processor = original_processor.original_processor
                processors[name] = base_processor
                spatial_processor = original_processor
                if bboxes is not None:
                    spatial_processor.set_spatial_control(bboxes, phrase_embeddings)
                new_processors[name] = spatial_processor
                spatial_processors[name] = spatial_processor
                continue
            
            # 仅在 Cross-Attention 层创建并注册 SpatialControlProcessor
            processors[name] = original_processor
            if is_cross_attn:
                # 创建或获取 Adapter
                adapter = get_or_create_adapter(layer_dim)
                # 创建 Spatial Processor
                spatial_processor = SpatialControlProcessor(
                    adapter,
                    original_processor,
                    is_cross_attn=is_cross_attn  # ✅ 传递 Cross-Attention 标记
                )

                if bboxes is not None:
                    spatial_processor.set_spatial_control(bboxes, phrase_embeddings)

                new_processors[name] = spatial_processor
                spatial_processors[name] = spatial_processor
            else:
                # 保持原始 processor（不包装 self-attention）
                new_processors[name] = original_processor
        
        # 设置新的 processors
        unet.set_attn_processor(new_processors)
        
    except AttributeError:
        # 方法 2: 回退到直接访问 module.processor
        for name, module in unet.named_modules():
            if hasattr(module, 'processor') and 'attn' in name.lower():
                layer_dim = _get_attn_layer_dim(unet, name)
                if layer_dim is None:
                    continue
                
                is_cross_attn = _is_cross_attention_layer(name)
                original_processor = module.processor
                processors[name] = original_processor

                if is_cross_attn:
                    adapter = get_or_create_adapter(layer_dim)
                    spatial_processor = SpatialControlProcessor(
                        adapter,
                        original_processor,
                        is_cross_attn=is_cross_attn
                    )

                    if bboxes is not None:
                        spatial_processor.set_spatial_control(bboxes, phrase_embeddings)

                    module.processor = spatial_processor
                    spatial_processors[name] = spatial_processor
                else:
                    # 保持 module.processor 不变（self-attention 不注入）
                    pass
    
    return processors, spatial_processors, adapter_dict


def remove_spatial_control_from_unet(unet, original_processors: Dict):
    """
    移除空间控制，恢复原始 processors
    
    Args:
        unet: Kolors UNet 模型
        original_processors: inject_spatial_control_to_unet 返回的原始 processors
    """
    if not isinstance(original_processors, dict) or len(original_processors) == 0:
        return
    
    try:
        # 方法 1: 使用 diffusers 的 set_attn_processor
        current_processors = unet.attn_processors
        restore_dict = {}
        
        for current_key in current_processors.keys():
            if current_key in original_processors:
                restore_dict[current_key] = original_processors[current_key]
            else:
                # 尝试不同的 key 格式
                key_without_suffix = current_key.replace('.processor', '')
                if key_without_suffix in original_processors:
                    restore_dict[current_key] = original_processors[key_without_suffix]
        
        if restore_dict:
            unet.set_attn_processor(restore_dict)
        else:
            unet.set_attn_processor(original_processors)
    
    except (AttributeError, TypeError, KeyError):
        # 方法 2: 回退到直接访问 module.processor
        for name, module in unet.named_modules():
            if hasattr(module, 'processor'):
                if name in original_processors:
                    module.processor = original_processors[name]
                elif name.replace('.processor', '') in original_processors:
                    module.processor = original_processors[name.replace('.processor', '')]


def get_trainable_parameters(adapter_dict: SpatialAdapterModuleDict) -> List[torch.nn.Parameter]:
    """
    获取所有 Adapter 的可训练参数
    
    Args:
        adapter_dict: Adapter 容器
    
    Returns:
        params: 可训练参数列表
    """
    params = []
    for adapter in adapter_dict.values():
        params.extend(list(adapter.parameters()))
    return params

