"""
评估指标模块

实现论文实验所需的各种评估指标：
1. 布局准确率（Layout Accuracy）
2. 生成质量（FID, CLIP Score）
3. 反馈修正效果
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
from scipy import linalg


def calculate_iou(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    计算两个 bounding box 的 IoU (Intersection over Union)
    
    Args:
        bbox1: (N, 4) [x1, y1, x2, y2] 归一化坐标
        bbox2: (M, 4) [x1, y1, x2, y2] 归一化坐标
    
    Returns:
        iou: (N, M) IoU 矩阵
    """
    # 扩展维度以便广播
    bbox1 = bbox1.unsqueeze(1)  # (N, 1, 4)
    bbox2 = bbox2.unsqueeze(0)  # (1, M, 4)
    
    # 计算交集
    x1_max = torch.max(bbox1[..., 0], bbox2[..., 0])  # (N, M)
    y1_max = torch.max(bbox1[..., 1], bbox2[..., 1])  # (N, M)
    x2_min = torch.min(bbox1[..., 2], bbox2[..., 2])  # (N, M)
    y2_min = torch.min(bbox1[..., 3], bbox2[..., 3])  # (N, M)
    
    # 交集面积（如果无交集则为0）
    inter_width = torch.clamp(x2_min - x1_max, min=0)
    inter_height = torch.clamp(y2_min - y1_max, min=0)
    inter_area = inter_width * inter_height  # (N, M)
    
    # 并集面积
    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])  # (N, 1)
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])  # (1, M)
    union_area = area1 + area2 - inter_area  # (N, M)
    
    # IoU
    iou = inter_area / (union_area + 1e-8)
    
    return iou


def layout_accuracy(predicted_bboxes: List[torch.Tensor],
                   target_bboxes: List[torch.Tensor],
                   iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    计算布局准确率
    
    Args:
        predicted_bboxes: 预测的 bbox 列表，每个元素是 (N_i, 4)
        target_bboxes: 真实的 bbox 列表，每个元素是 (M_i, 4)
        iou_threshold: IoU 阈值（超过此值认为匹配成功）
    
    Returns:
        {
            "mean_iou": float,  # 平均 IoU
            "match_rate": float,  # 匹配率（IoU > threshold 的比例）
            "position_accuracy": float,  # 位置准确率（基于方向匹配）
        }
    """
    all_ious = []
    all_matches = []
    
    for pred_bbox, target_bbox in zip(predicted_bboxes, target_bboxes):
        if len(pred_bbox) == 0 or len(target_bbox) == 0:
            continue
        
        # 转换为 tensor（如果还不是）
        if not isinstance(pred_bbox, torch.Tensor):
            pred_bbox = torch.tensor(pred_bbox, dtype=torch.float32)
        if not isinstance(target_bbox, torch.Tensor):
            target_bbox = torch.tensor(target_bbox, dtype=torch.float32)
        
        # 计算 IoU 矩阵
        iou_matrix = calculate_iou(pred_bbox, target_bbox)  # (N, M)
        
        # 找到最佳匹配（匈牙利算法简化版：贪心匹配）
        matches = []
        used_targets = set()
        
        # 按 IoU 降序排序
        flat_indices = torch.argsort(iou_matrix.flatten(), descending=True)
        
        for idx in flat_indices:
            n_idx = idx.item() // iou_matrix.shape[1]
            m_idx = idx.item() % iou_matrix.shape[1]
            
            if m_idx not in used_targets and iou_matrix[n_idx, m_idx] > iou_threshold:
                matches.append((n_idx, m_idx, iou_matrix[n_idx, m_idx].item()))
                used_targets.add(m_idx)
        
        # 记录匹配的 IoU
        for _, _, iou in matches:
            all_ious.append(iou)
            all_matches.append(1.0)
        
        # 记录未匹配的（IoU = 0）
        num_unmatched = len(pred_bbox) - len(matches)
        all_ious.extend([0.0] * num_unmatched)
        all_matches.extend([0.0] * num_unmatched)
    
    if len(all_ious) == 0:
        return {
            "mean_iou": 0.0,
            "match_rate": 0.0,
            "position_accuracy": 0.0
        }
    
    mean_iou = np.mean(all_ious)
    match_rate = np.mean(all_matches)
    
    return {
        "mean_iou": float(mean_iou),
        "match_rate": float(match_rate),
        "position_accuracy": float(match_rate)  # 简化：使用 match_rate 作为位置准确率
    }


def position_description_accuracy(predicted_bboxes: List[torch.Tensor],
                                 captions: List[str],
                                 position_keywords: Dict[str, Tuple[float, float, float, float]]) -> float:
    """
    计算位置描述匹配度（"左边"是否真的在左边）
    
    Args:
        predicted_bboxes: 预测的 bbox 列表
        captions: 对应的 caption 列表
        position_keywords: 位置关键词到标准 bbox 的映射
            {"左边": [0.0, 0.0, 0.5, 1.0], ...}
    
    Returns:
        accuracy: 位置描述准确率
    """
    correct_count = 0
    total_count = 0
    
    for bbox, caption in zip(predicted_bboxes, captions):
        if len(bbox) == 0:
            continue
        
        # 转换为 tensor
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        
        # 检查 caption 中的位置关键词
        for keyword, expected_bbox in position_keywords.items():
            if keyword in caption:
                # 计算预测 bbox 与期望 bbox 的 IoU
                expected_bbox_tensor = torch.tensor(expected_bbox, dtype=torch.float32).unsqueeze(0)
                iou = calculate_iou(bbox, expected_bbox_tensor)  # (N, 1)
                max_iou = iou.max().item()
                
                if max_iou > 0.3:  # 阈值可调
                    correct_count += 1
                total_count += 1
                break
    
    if total_count == 0:
        return 0.0
    
    return correct_count / total_count


def clip_score(images: List[Image.Image],
              texts: List[str],
              clip_model=None,
              processor=None) -> float:
    """
    计算 CLIP Score（图像-文本相似度）
    
    Args:
        images: 图像列表
        texts: 文本列表
        clip_model: CLIP 模型（如果为 None，会自动加载）
        processor: CLIP processor（如果为 None，会自动加载）
    
    Returns:
        mean_score: 平均 CLIP Score
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        
        if clip_model is None or processor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        device = next(clip_model.parameters()).device
        clip_model.eval()
        
        scores = []
        with torch.no_grad():
            for image, text in zip(images, texts):
                inputs = processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                score = logits_per_image.softmax(dim=1)[0, 0].item()
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
        
    except Exception as e:
        print(f"⚠️ CLIP Score 计算出错: {e}")
        return 0.0


def calculate_fid(real_images: List[np.ndarray],
                 generated_images: List[np.ndarray],
                 batch_size: int = 50) -> float:
    """
    计算 FID (Fréchet Inception Distance)
    
    Args:
        real_images: 真实图像列表（numpy array，已归一化到 [0, 1]）
        generated_images: 生成图像列表
        batch_size: 批处理大小
    
    Returns:
        fid_score: FID 分数（越小越好）
    """
    try:
        from torchvision.models import inception_v3
        import torchvision.transforms as transforms
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载 Inception v3
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        
        # 预处理
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        def get_features(images):
            features = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                batch_tensors = torch.stack([transform(img) for img in batch]).to(device)
                
                with torch.no_grad():
                    batch_features = inception(batch_tensors)
                    features.append(batch_features.cpu().numpy())
            
            return np.concatenate(features, axis=0)
        
        # 提取特征
        real_features = get_features(real_images)
        gen_features = get_features(generated_images)
        
        # 计算 FID
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        
        return float(fid)
        
    except Exception as e:
        print(f"⚠️ FID 计算出错: {e}")
        return float('inf')


def feedback_correction_metrics(before_results: List[Dict],
                               after_results: List[Dict]) -> Dict[str, float]:
    """
    计算反馈修正效果
    
    Args:
        before_results: 修正前的验证结果列表
        after_results: 修正后的验证结果列表
    
    Returns:
        {
            "accuracy_improvement": float,  # 准确率提升
            "mean_correction_rounds": float,  # 平均修正轮次
            "success_rate": float,  # 修正成功率
        }
    """
    before_correct = sum(1 for r in before_results if r.get("correct", False))
    after_correct = sum(1 for r in after_results if r.get("correct", False))
    
    before_accuracy = before_correct / len(before_results) if before_results else 0.0
    after_accuracy = after_correct / len(after_results) if after_results else 0.0
    
    accuracy_improvement = after_accuracy - before_accuracy
    
    # 计算修正轮次（简化：假设每次修正需要1轮）
    correction_rounds = []
    for before_r, after_r in zip(before_results, after_results):
        if not before_r.get("correct", False) and after_r.get("correct", False):
            correction_rounds.append(1)  # 成功修正
        elif not before_r.get("correct", False):
            correction_rounds.append(2)  # 修正失败（假设尝试了2次）
    
    mean_correction_rounds = np.mean(correction_rounds) if correction_rounds else 0.0
    
    # 修正成功率
    success_count = sum(1 for before_r, after_r in zip(before_results, after_results)
                      if not before_r.get("correct", False) and after_r.get("correct", False))
    total_attempts = sum(1 for r in before_results if not r.get("correct", False))
    success_rate = success_count / total_attempts if total_attempts > 0 else 0.0
    
    return {
        "accuracy_improvement": float(accuracy_improvement),
        "mean_correction_rounds": float(mean_correction_rounds),
        "success_rate": float(success_rate)
    }


def comprehensive_evaluation(predicted_bboxes: List[torch.Tensor],
                            target_bboxes: List[torch.Tensor],
                            generated_images: List[Image.Image],
                            captions: List[str],
                            before_feedback: Optional[List[Dict]] = None,
                            after_feedback: Optional[List[Dict]] = None) -> Dict[str, float]:
    """
    综合评估（论文实验用）
    
    Args:
        predicted_bboxes: 预测的 bbox
        target_bboxes: 真实的 bbox
        generated_images: 生成的图像
        captions: 对应的 caption
        before_feedback: 修正前的反馈结果（可选）
        after_feedback: 修正后的反馈结果（可选）
    
    Returns:
        包含所有评估指标的字典
    """
    results = {}
    
    # 1. 布局准确率
    layout_metrics = layout_accuracy(predicted_bboxes, target_bboxes)
    results.update({f"layout_{k}": v for k, v in layout_metrics.items()})
    
    # 2. 位置描述准确率
    position_keywords = {
        "左边": [0.0, 0.0, 0.5, 1.0],
        "右边": [0.5, 0.0, 1.0, 1.0],
        "中间": [0.25, 0.25, 0.75, 0.75],
        "上方": [0.0, 0.0, 1.0, 0.5],
        "下方": [0.0, 0.5, 1.0, 1.0],
    }
    results["position_description_accuracy"] = position_description_accuracy(
        predicted_bboxes, captions, position_keywords
    )
    
    # 3. CLIP Score
    results["clip_score"] = clip_score(generated_images, captions)
    
    # 4. 反馈修正效果（如果有）
    if before_feedback and after_feedback:
        feedback_metrics = feedback_correction_metrics(before_feedback, after_feedback)
        results.update({f"feedback_{k}": v for k, v in feedback_metrics.items()})
    
    return results

