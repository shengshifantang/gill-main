"""Helper file defining some common loss functions."""
from typing import Optional
import torch
from gill import utils


def l1_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Args:
    u: (N, D) tensor.
    v: (N, D) tensor.
  Returns:
    l1_loss: (N,) tensor of summed L1 loss.
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return torch.abs(u - v).sum(dim=-1)


def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Compute L2 (Euclidean) distance between two tensors.
  
  Args:
    u: (N, T, D) tensor.
    v: (N, T, D) tensor.
  Returns:
    l2_distance: (N, T) tensor of L2 distances (sqrt of sum of squared differences).
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return ((u - v) ** 2).sum(dim=-1) ** 0.5


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def contrastive_acc(logits: torch.Tensor, target: Optional[torch.Tensor] = None, topk=(1,)) -> torch.Tensor:
  """
  Args:
    logits: (N, N) predictions.
    target: (N, num_correct_answers) labels.
  """
  assert len(logits.shape) == 2, logits.shape
  batch_size = logits.shape[0]

  if target is None:
    target = torch.arange(len(logits), device=logits.device)
    return utils.accuracy(logits, target, -1, topk)
  else:
    assert len(target.shape) == 2, target.shape
    with torch.no_grad():
      maxk = max(topk)
      if logits.shape[-1] < maxk:
        print(f"[WARNING] Less than {maxk} predictions available. Using {logits.shape[-1]} for topk.")
      maxk = min(maxk, logits.shape[-1])

      # Take topk along the last dimension.
      _, pred = logits.topk(maxk, -1, True, True)  # (N, topk)
      assert pred.shape == (batch_size, maxk)

      target_expand = target[:, :, None].repeat(1, 1, maxk)  # (N, num_correct_answers, topk)
      pred_expand = pred[:, None, :].repeat(1, target.shape[1], 1)  # (N, num_correct_answers, topk)
      correct = pred_expand.eq(target_expand)  # (N, num_correct_answers, topk)
      correct = torch.any(correct, dim=1)  # (N, topk)

      res = []
      for k in topk:
        any_k_correct = torch.clamp(correct[:, :k].sum(1), max=1)  # (N,)
        correct_k = any_k_correct.float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
      return res

def layout_loss(predicted_bboxes: torch.Tensor, target_bboxes: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
  """
  计算布局损失（L1 损失）
  
  Args:
    predicted_bboxes: (B, N, 4) 预测的 bounding box [x1, y1, x2, y2]
    target_bboxes: (B, M, 4) 真实的 bounding box
    mask: (B, N) 可选，标记哪些预测是有效的
  
  Returns:
    loss: 标量损失值
  """
  B, N, _ = predicted_bboxes.shape
  _, M, _ = target_bboxes.shape
  
  # 如果数量不匹配，需要匹配（简单的最近邻匹配）
  if N != M:
    # 计算所有预测和真实 bbox 之间的距离
    pred_flat = predicted_bboxes.view(B * N, 4)  # (B*N, 4)
    target_flat = target_bboxes.view(B * M, 4)  # (B*M, 4)
    
    # L1 距离
    distances = torch.abs(pred_flat.unsqueeze(1) - target_flat.unsqueeze(0)).sum(dim=-1)  # (B*N, B*M)
    distances = distances.view(B, N, B, M)
    
    # 为每个样本找到最佳匹配
    matched_targets = []
    for b in range(B):
      # 找到最小距离匹配
      sample_distances = distances[b, :, b, :]  # (N, M)
      _, matched_indices = torch.min(sample_distances, dim=1)  # (N,)
      matched_target = target_bboxes[b][matched_indices]  # (N, 4)
      matched_targets.append(matched_target)
    
    target_bboxes = torch.stack(matched_targets, dim=0)  # (B, N, 4)
  
  # 计算 L1 损失
  loss = torch.abs(predicted_bboxes - target_bboxes).sum(dim=-1)  # (B, N)
  
  # 应用 mask（如果有）
  if mask is not None:
    loss = loss * mask
    loss = loss.sum() / (mask.sum() + 1e-8)
  else:
    loss = loss.mean()
  
  return loss


def feedback_loss(verification_results: torch.Tensor, 
                 correction_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
  """
  基于验证结果的反馈损失（强化学习风格）
  
  Args:
    verification_results: (B,) 验证结果（1.0 表示正确，0.0 表示错误）
    correction_weights: (B,) 可选，修正权重
  
  Returns:
    loss: 标量损失值
  """
  # 将验证结果转换为损失（错误越多，损失越大）
  loss = 1.0 - verification_results  # (B,)
  
  # 应用权重（如果有）
  if correction_weights is not None:
    loss = loss * correction_weights
  
  return loss.mean()

