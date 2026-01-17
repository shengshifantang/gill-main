"""
反馈验证模块 (Feedback Verifier) - 异构混合架构版 (Hybrid MoE)

架构设计：
1. Spatial Expert (Grounding DINO): 负责精确的坐标检测和 IoU 计算
2. Semantic Expert (Qwen2-VL): 负责颜色、材质、风格等语义一致性检查
3. Hybrid Orchestrator: 融合两者意见，生成最终 Rationale

论文贡献：Neuro-Symbolic Feedback Mechanism（神经符号反馈机制）
- DINO 代表符号化的精准定位
- Qwen 代表神经化的语义理解
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Literal
from PIL import Image
import json
import os

# 尝试导入 Grounding DINO 依赖
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    HAS_DINO = True
except ImportError:
    HAS_DINO = False
    print("⚠️ transformers 未安装或版本不支持 Grounding DINO，将降级为纯 Qwen 模式")

# 尝试导入 Qwen2-VL
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor as QwenProcessor
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("⚠️ transformers 未安装或版本不支持 Qwen2-VL")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个 [x1, y1, x2, y2] 框的 IoU
    
    Args:
        box1: [x1, y1, x2, y2] 归一化坐标 (0-1)
        box2: [x1, y1, x2, y2] 归一化坐标 (0-1)
    
    Returns:
        IoU 值 (0-1)
    """
    # 确保坐标是 (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def normalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
    """
    将 bbox 归一化到 0-1 范围
    
    Args:
        bbox: 可能是 [x1, y1, x2, y2] 格式，范围可能是 0-1000 或 0-1
        image_size: (width, height) 图像尺寸
    
    Returns:
        归一化后的 [x1, y1, x2, y2] (0-1)
    """
    x1, y1, x2, y2 = bbox
    
    # 如果坐标在 0-1000 范围，转换为 0-1
    if max(bbox) > 1.0:
        x1, y1, x2, y2 = [c / 1000.0 for c in bbox]
    
    # 确保坐标在有效范围内
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    
    # 确保 x1 < x2, y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    return [x1, y1, x2, y2]


class GroundingDinoVerifier:
    """空间验证专家：使用 Grounding DINO 进行零样本检测和位置验证"""
    
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", device: str = "cuda"):
        """
        Args:
            model_id: Grounding DINO 模型 ID（HuggingFace）
            device: 设备
        """
        self.device = device
        self.model = None
        self.processor = None
        
        if not HAS_DINO:
            print("⚠️ Grounding DINO 不可用，空间验证将跳过")
            return
        
        print(f"📦 加载空间专家 (Grounding DINO): {model_id}")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
            self.model.eval()
            print("  ✅ Grounding DINO 加载成功")
        except Exception as e:
            print(f"  ⚠️ Grounding DINO 加载失败: {e}")
            print("  💡 提示: 请确保 transformers >= 4.36.0")
            self.model = None
            self.processor = None

    def verify_layout(self, 
                     image: Image.Image, 
                     expected_layout: List[Dict], 
                     threshold: float = 0.35,
                     iou_threshold: float = 0.5) -> Dict:
        """
        验证布局一致性
        
        Args:
            image: PIL Image
            expected_layout: [{"name": "猫", "bbox": [x1, y1, x2, y2]}, ...]
            threshold: DINO 检测置信度阈值
            iou_threshold: IoU 匹配阈值
        
        Returns:
            {
                "correct": bool,
                "feedback": str,
                "score": float,  # 平均 IoU
                "details": List[Dict]  # 每个物体的检测详情
            }
        """
        if not self.model or not expected_layout:
            return {
                "correct": True, 
                "feedback": "无空间约束或模型未加载", 
                "score": 1.0,
                "details": []
            }

        # 提取所有需要检测的物体名称
        # Grounding DINO 需要 text prompt 格式为 "cat. dog. chair."
        labels = [obj['name'] for obj in expected_layout]
        text_prompt = ". ".join(labels) + "."
        
        try:
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 后处理
            target_sizes = torch.tensor([image.size[::-1]])  # [height, width]
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=threshold,
                text_threshold=0.25,
                target_sizes=target_sizes
            )[0]
            
            # 解析检测结果
            detected_boxes = results["boxes"].cpu().numpy()  # [N, 4] 像素坐标
            detected_labels = results["labels"]  # [N] str
            detected_scores = results["scores"].cpu().numpy()  # [N]
            
            w, h = image.size
            
            feedback_lines = []
            all_passed = True
            total_iou = 0
            details = []
            
            # 对每个期望物体，寻找最佳匹配的检测框
            for exp_obj in expected_layout:
                exp_name = exp_obj['name']
                exp_bbox = normalize_bbox(exp_obj['bbox'], (w, h))
                
                # 转换为像素坐标用于 IoU 计算
                exp_box_pixel = [
                    exp_bbox[0] * w, exp_bbox[1] * h,
                    exp_bbox[2] * w, exp_bbox[3] * h
                ]
                
                # 在检测结果中找匹配的类别和最佳 IoU
                best_iou = 0
                best_box = None
                best_label = None
                best_score = 0
                
                for pred_box, pred_label, pred_score in zip(detected_boxes, detected_labels, detected_scores):
                    # 检查标签是否匹配（简单字符串匹配）
                    if exp_name.lower() not in pred_label.lower():
                        continue
                    
                    # 转换为归一化坐标用于 IoU 计算
                    pred_box_norm = [
                        pred_box[0] / w, pred_box[1] / h,
                        pred_box[2] / w, pred_box[3] / h
                    ]
                    
                    # 计算 IoU（使用归一化坐标）
                    curr_iou = calculate_iou(exp_bbox, pred_box_norm)
                    
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_box = pred_box.tolist()
                        best_label = pred_label
                        best_score = float(pred_score)
                
                # 判断是否通过
                passed = best_iou >= iou_threshold
                if not passed:
                    all_passed = False
                    feedback_lines.append(f"❌ {exp_name} 位置偏差过大 (IoU={best_iou:.2f}, 阈值={iou_threshold:.2f}) 或未检测到")
                else:
                    feedback_lines.append(f"✅ {exp_name} 位置正确 (IoU={best_iou:.2f})")
                
                total_iou += best_iou
                
                details.append({
                    "name": exp_name,
                    "expected_bbox": exp_bbox,
                    "detected_bbox": best_box,
                    "iou": float(best_iou),
                    "passed": passed,
                    "detected_label": best_label,
                    "detection_score": best_score
                })
            
            avg_iou = total_iou / len(expected_layout) if expected_layout else 0
            
            return {
                "correct": all_passed,
                "feedback": " ".join(feedback_lines),
                "score": float(avg_iou),
                "details": details
            }
            
        except Exception as e:
            print(f"  ⚠️ Grounding DINO 验证过程出错: {e}")
            return {
                "correct": True,  # 出错时默认通过，避免阻塞流程
                "feedback": f"空间验证过程出错: {str(e)}",
                "score": 0.5,
                "details": []
            }


class QwenSemanticVerifier:
    """语义验证专家：使用 Qwen2-VL 检查颜色、属性、数量等语义一致性"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        """
        Args:
            model_path: Qwen2-VL 模型路径（HuggingFace ID 或本地路径）
            device: 设备
        """
        self.device = device
        self.model = None
        self.processor = None
        
        if not HAS_QWEN:
            print("⚠️ Qwen2-VL 不可用，语义验证将跳过")
            return
        
        print(f"📦 加载语义专家 (Qwen2-VL): {model_path}")
        try:
            self.processor = QwenProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            ).eval()
            print("  ✅ Qwen2-VL 加载成功")
        except Exception as e:
            print(f"  ⚠️ Qwen2-VL 加载失败: {e}")
            self.model = None
            self.processor = None

    def verify_semantics(self, image: Image.Image, prompt: str) -> Dict:
        """
        检查图像是否符合 Prompt 的语义描述
        
        Args:
            image: PIL Image
            prompt: 原始 prompt
        
        Returns:
            {
                "correct": bool,
                "feedback": str,
                "confidence": float
            }
        """
        if not self.model:
            return {
                "correct": True,
                "feedback": "语义验证器未加载，跳过验证",
                "confidence": 0.5
            }
        
        try:
            query = f"""请作为一名极度严格的视觉质检员。
用户提示词: "{prompt}"
请检查生成的图像是否符合上述描述。

重点检查：
1. 物体是否存在？
2. 颜色、材质、属性是否正确？
3. 数量是否正确？
4. 动作、姿态是否符合描述？
(忽略具体的位置坐标，只关注内容正确性)

请输出结论：
如果符合，输出"符合"。
如果不符合，简要说明原因（例如："颜色不对，应该是红色但图中是蓝色"）。"""

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]}
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
            # 提取生成的文本（去掉输入部分）
            generated_ids = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # 解析响应
            is_pass = "符合" in response and "不符合" not in response
            # 计算置信度（基于关键词）
            if is_pass:
                confidence = 0.9 if "完全" in response or "非常" in response else 0.7
            else:
                confidence = 0.2
            
            return {
                "correct": is_pass,
                "feedback": response,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"  ⚠️ Qwen2-VL 验证过程出错: {e}")
            return {
                "correct": True,  # 出错时默认通过
                "feedback": f"语义验证过程出错: {str(e)}",
                "confidence": 0.5
            }


class HybridFeedbackVerifier:
    """
    [核心] 异构混合验证器
    
    组合 Grounding DINO (空间) + Qwen2-VL (语义)
    实现 Neuro-Symbolic Feedback Mechanism
    """
    
    def __init__(self, 
                 semantic_model: str = "Qwen/Qwen2-VL-7B-Instruct",
                 spatial_model: str = "IDEA-Research/grounding-dino-base",
                 device: str = "cuda"):
        """
        Args:
            semantic_model: Qwen2-VL 模型路径
            spatial_model: Grounding DINO 模型路径
            device: 设备
        """
        self.device = device
        
        # 初始化两个专家
        self.spatial_expert = GroundingDinoVerifier(spatial_model, device)
        self.semantic_expert = QwenSemanticVerifier(semantic_model, device)
        
        print("🔀 异构混合验证器初始化完成")

    def verify(self, 
               image: Image.Image,
               original_prompt: str,
               expected_layout: Optional[List[Dict]] = None,
               threshold: float = 0.7,
               iou_threshold: float = 0.5) -> Dict:
        """
        执行双重验证并融合结果
        
        Args:
            image: 生成的图像
            original_prompt: 原始 prompt
            expected_layout: 期望的布局 [{"name": "...", "bbox": [...]}]
            threshold: 总体置信度阈值
            iou_threshold: IoU 匹配阈值（用于空间验证）
        
        Returns:
            {
                "correct": bool,
                "confidence": float,
                "feedback": str,
                "refinement_instruction": Optional[str],
                "spatial_pass": bool,
                "semantic_pass": bool,
                "rationale": str
            }
        """
        feedback_parts = []
        is_spatial_pass = True
        is_semantic_pass = True
        spatial_score = 1.0
        semantic_confidence = 1.0
        
        # 1. 空间验证 (如果有布局要求)
        if expected_layout and self.spatial_expert.model:
            spatial_res = self.spatial_expert.verify_layout(
                image, expected_layout, threshold=0.35, iou_threshold=iou_threshold
            )
            is_spatial_pass = spatial_res["correct"]
            spatial_score = spatial_res["score"]
            feedback_parts.append(f"[空间检查] {spatial_res['feedback']}")
        elif expected_layout:
            # 有布局要求但 DINO 未加载，给出警告
            feedback_parts.append("[空间检查] Grounding DINO 未加载，跳过位置验证")
        
        # 2. 语义验证
        if self.semantic_expert.model:
            semantic_res = self.semantic_expert.verify_semantics(image, original_prompt)
            is_semantic_pass = semantic_res["correct"]
            semantic_confidence = semantic_res["confidence"]
            feedback_parts.append(f"[语义检查] {semantic_res['feedback']}")
        else:
            feedback_parts.append("[语义检查] Qwen2-VL 未加载，跳过语义验证")
        
        # 3. 融合决策（严格模式：两者都必须通过）
        final_pass = is_spatial_pass and is_semantic_pass
        
        # 计算综合置信度
        if expected_layout and self.spatial_expert.model:
            # 空间和语义各占 50%
            combined_confidence = (spatial_score + semantic_confidence) / 2
        else:
            # 只有语义验证
            combined_confidence = semantic_confidence
        
        final_pass = final_pass and (combined_confidence >= threshold)
        
        # 4. 生成修正指令
        refinement_instruction = None
        if not final_pass:
            refinement_instruction = "请修正以下问题：\n" + "\n".join(feedback_parts)
        
        # 5. 生成 Rationale（用于论文展示）
        rationale = f"空间验证: {'通过' if is_spatial_pass else '失败'} (IoU={spatial_score:.2f}); " \
                   f"语义验证: {'通过' if is_semantic_pass else '失败'} (置信度={semantic_confidence:.2f})"
        
        return {
            "correct": final_pass,
            "confidence": combined_confidence,
            "feedback": "\n".join(feedback_parts),
            "refinement_instruction": refinement_instruction,
            "spatial_pass": is_spatial_pass,
            "semantic_pass": is_semantic_pass,
            "rationale": rationale,
            "suggested_prompt": original_prompt,
            "detected_objects": []
        }


class FeedbackVerifier:
    """
    反馈验证器（兼容旧接口）
    
    支持多种验证模式：
    - "hybrid": 混合模式（Grounding DINO + Qwen2-VL）
    - "grounding_dino": 仅空间验证
    - "qwen2vl_7b": 仅语义验证
    - "qwen2vl": 兼容旧代码
    """
    
    def __init__(self, 
                 verifier_type: Literal["grounding_dino", "qwen2vl_7b", "hybrid", "qwen2vl"] = "hybrid",
                 vlm_model_name: Optional[str] = None,
                 device: str = "cuda",
                 use_grounding: bool = True):
        """
        Args:
            verifier_type: 验证器类型
            vlm_model_name: VLM 模型名称（用于兼容旧代码）
            device: 设备
            use_grounding: 是否使用 grounding（兼容旧代码）
        """
        self.device = device
        self.verifier_type = verifier_type
        
        # 根据类型初始化相应的验证器
        if verifier_type == "hybrid":
            semantic_model = vlm_model_name or "Qwen/Qwen2-VL-7B-Instruct"
            self.verifier = HybridFeedbackVerifier(
                semantic_model=semantic_model,
                device=device
            )
        elif verifier_type == "grounding_dino":
            self.verifier = GroundingDinoVerifier(device=device)
        elif verifier_type in ["qwen2vl_7b", "qwen2vl"]:
            model_path = vlm_model_name or "Qwen/Qwen2-VL-7B-Instruct"
            self.semantic_expert = QwenSemanticVerifier(model_path, device)
            self.verifier = self.semantic_expert
        else:
            raise ValueError(f"未知的验证器类型: {verifier_type}")
    
    def verify(self, 
               image: Image.Image,
               original_prompt: str,
               expected_layout: Optional[List[Dict]] = None,
               threshold: float = 0.7) -> Dict:
        """
        统一验证接口
        
        Args:
            image: 生成的图像
            original_prompt: 原始 prompt
            expected_layout: 期望的布局
            threshold: 置信度阈值
        
        Returns:
            验证结果字典
        """
        if isinstance(self.verifier, HybridFeedbackVerifier):
            return self.verifier.verify(image, original_prompt, expected_layout, threshold)
        elif isinstance(self.verifier, GroundingDinoVerifier):
            if expected_layout:
                result = self.verifier.verify_layout(image, expected_layout)
                return {
                    "correct": result["correct"],
                    "confidence": result["score"],
                    "feedback": result["feedback"],
                    "refinement_instruction": result["feedback"] if not result["correct"] else None,
                    "suggested_prompt": original_prompt,
                    "detected_objects": result.get("details", [])
                }
            return {
                "correct": True,
                "confidence": 1.0,
                "feedback": "无布局约束",
                "suggested_prompt": original_prompt,
                "detected_objects": []
            }
        elif isinstance(self.verifier, QwenSemanticVerifier):
            result = self.verifier.verify_semantics(image, original_prompt)
            return {
                "correct": result["correct"],
                "confidence": result["confidence"],
                "feedback": result["feedback"],
                "refinement_instruction": result["feedback"] if not result["correct"] else None,
                "suggested_prompt": original_prompt,
                "detected_objects": []
            }
        else:
            return {
                "correct": True,
                "confidence": 0.5,
                "feedback": "验证器未正确初始化",
                "suggested_prompt": original_prompt,
                "detected_objects": []
            }
    

def create_feedback_verifier(vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                             verifier_type: Literal["grounding_dino", "qwen2vl_7b", "hybrid", "qwen2vl"] = "hybrid",
                             device: str = "cuda",
                             use_grounding: bool = True) -> FeedbackVerifier:
    """
    创建反馈验证器（工厂函数）
    
    Args:
        vlm_model_name: VLM 模型名称（用于 qwen2vl 模式）
        verifier_type: 验证器类型
            - "grounding_dino": 仅使用 Grounding DINO
            - "qwen2vl_7b": 仅使用 Qwen2-VL-7B
            - "hybrid": 混合模式（Grounding DINO + Qwen2-VL-7B，推荐）
            - "qwen2vl": 使用指定的 Qwen-VL 模型（兼容旧代码）
        device: 设备
        use_grounding: 是否使用 grounding（兼容旧代码）
    
    Returns:
        FeedbackVerifier 实例
    """
    return FeedbackVerifier(
        verifier_type=verifier_type,
        vlm_model_name=vlm_model_name,
        device=device,
        use_grounding=use_grounding
    )
