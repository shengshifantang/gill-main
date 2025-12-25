import torch
from torch import nn
from typing import Tuple, Optional


class TextFcLayer(nn.Module):
  """Layers used in mapping text embeddings to visual outputs.
  
  支持两种模式:
  - 'linear': 简单线性变换
  - 'gill_mapper': 使用 Transformer 的复杂映射器
  
  支持输出 pooled embedding (用于 Kolors 等需要 pooled_prompt_embeds 的模型)
  """

  def __init__(self, in_dim: int, out_dim: int, num_input_tokens: int = 1, 
               num_output_tokens: int = 1, mode: str = 'linear',
               output_pooled: bool = False):
    """
    Args:
      in_dim: 输入维度 (e.g., LLM hidden_size = 4096)
      out_dim: 输出维度 (e.g., 768 for SD, 2048 for Kolors)
      num_input_tokens: 输入 token 数量
      num_output_tokens: 输出 token 数量 (e.g., 77 for SD, 256 for Kolors)
      mode: 'linear' 或 'gill_mapper'
      output_pooled: 是否同时输出 pooled embedding (用于 Kolors)
    """
    super().__init__()

    self.num_input_tokens = num_input_tokens
    self.num_output_tokens = num_output_tokens
    self.mode = mode
    self.out_dim = out_dim
    self.output_pooled = output_pooled

    if mode == 'linear':
      self.model = nn.Linear(in_dim, out_dim)
    elif mode == 'gill_mapper':
      # 根据输出维度调整 hidden_dim
      # Kolors (out_dim=2048) 需要更大的 hidden_dim
      hidden_dim = 512 if out_dim <= 768 else 1024
      self.fc = nn.Linear(in_dim, hidden_dim)
      self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
      self.model = nn.Linear(hidden_dim, out_dim)
      self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))
    else:
      raise NotImplementedError(mode)
    
    # Pooled embedding 投影层 (用于 Kolors)
    if output_pooled:
      self.pooled_fc = nn.Linear(out_dim, out_dim)
      print(f"TextFcLayer: 启用 pooled embedding 输出 (dim={out_dim})")

  def forward(self, x: torch.Tensor, input_embs: torch.Tensor, 
              return_pooled: bool = False) -> torch.Tensor:
    """
    Forward pass for text-to-visual embedding mapping.
    
    Args:
      x: (N, T_in, D_in) input tensor (e.g., LLM hidden states)
      input_embs: (N, T_in, D_in) additional embeddings to add (for gill_mapper mode)
      return_pooled: 是否返回 pooled embedding (需要 output_pooled=True 初始化)
    
    Returns:
      如果 return_pooled=False:
        outputs: (N, T_out, D_out) mapped embeddings
      如果 return_pooled=True:
        (outputs, pooled): ((N, T_out, D_out), (N, D_out))
    """
    outputs = None
    
    if self.mode == 'gill_mapper':
      x = x + input_embs

    if isinstance(self.model, nn.ModuleList):
      assert len(self.model) == x.shape[1] == self.num_input_tokens, (len(self.model), x.shape, self.num_input_tokens)
      outputs = []
      for i in range(self.num_input_tokens):
        outputs.append(self.model[i](x[:, i, :]))  # (N, D)
      outputs = torch.stack(outputs, dim=1)  # (N, T, D)
    else:
      if self.mode == 'gill_mapper':
        x = self.fc(x)
        # 使用 expand 替代 repeat，避免不必要的内存复制
        query = self.query_embs.expand(x.shape[0], -1, -1)
        x = self.tfm(x, query)
      elif self.mode == 'linear':
        # Linear 模式：处理 token 数量不匹配的情况
        if self.num_input_tokens != self.num_output_tokens:
          # 使用平均池化后 expand 到目标 token 数
          x = x.mean(dim=1, keepdim=True)  # (N, 1, D_in)
          x = x.expand(-1, self.num_output_tokens, -1)  # (N, T_out, D_in)
      
      outputs = self.model(x)

      # 确保输出 token 数正确（仅在必要时裁剪）
      if outputs.shape[1] > self.num_output_tokens:
        outputs = outputs[:, :self.num_output_tokens, :]
    
    # 验证输出形状
    assert outputs.shape[1] == self.num_output_tokens or \
           (outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * outputs.shape[2]), \
           f"Output shape mismatch: {outputs.shape}, expected T={self.num_output_tokens}"
    
    # 生成 pooled embedding (用于 Kolors)
    if return_pooled and self.output_pooled:
      # 使用第一个 token 的输出作为 pooled embedding（类似 BERT [CLS]）
      # 或者使用平均池化
      pooled = outputs.mean(dim=1)  # (N, D_out)
      pooled = self.pooled_fc(pooled)  # (N, D_out)
      return outputs, pooled
    
    return outputs  # (N, T_out, D_out)

