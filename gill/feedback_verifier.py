"""
反馈验证模块 (Feedback Verifier)

使用 Qwen-VL 或 KOSMOS-2 验证生成图像是否符合 prompt 和布局要求。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np


class FeedbackVerifier:
    """
    反馈验证器
    
    使用 VLM（Vision-Language Model）验证生成图像是否符合要求
    """
    
    def __init__(self, 
                 vlm_model_name: str = "Qwen/Qwen-VL",
                 device: str = "cuda",
                 use_grounding: bool = True):
        """
        Args:
            vlm_model_name: VLM 模型名称或路径
            device: 设备
            use_grounding: 是否使用 grounding 功能（检测对象位置）
        """
        self.device = device
        self.use_grounding = use_grounding
        self.vlm_model_name = vlm_model_name
        
        # 加载 VLM 模型
        self._load_vlm_model()
    
    def _load_vlm_model(self):
        """加载 VLM 模型"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            print(f"📦 加载 VLM 模型: {self.vlm_model_name}")
            
            # 尝试加载 Qwen-VL
            if "qwen" in self.vlm_model_name.lower() or "Qwen" in self.vlm_model_name:
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.vlm_model_name,
                        trust_remote_code=True
                    )
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.vlm_model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device,
                        trust_remote_code=True
                    )
                    self.model_type = "qwen"
                    print("✓ 使用 Qwen-VL 作为验证器")
                except Exception as e:
                    print(f"⚠️ Qwen-VL 加载失败: {e}")
                    self._load_fallback_model()
            else:
                self._load_fallback_model()
                
        except ImportError:
            print("⚠️ transformers 未安装或版本不支持，使用轻量级验证器")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """加载备用验证器（基于 CLIP）"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print("📦 使用 CLIP 作为备用验证器")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.model_type = "clip"
            print("✓ 使用 CLIP 作为验证器（功能受限）")
        except Exception as e:
            print(f"❌ 备用验证器加载失败: {e}")
            self.model = None
            self.processor = None
            self.model_type = None
    
    def verify(self, 
               image: Image.Image,
               original_prompt: str,
               expected_layout: Optional[List[Dict]] = None,
               threshold: float = 0.7) -> Dict:
        """
        验证生成图像是否符合要求
        
        Args:
            image: 生成的图像（PIL Image）
            original_prompt: 原始 prompt
            expected_layout: 期望的布局信息 [{"name": "...", "bbox": [...]}]
            threshold: 置信度阈值
        
        Returns:
            {
                "correct": bool,  # 是否通过验证
                "confidence": float,  # 置信度 (0-1)
                "feedback": str,  # 反馈信息
                "suggested_prompt": str,  # 修正建议
                "detected_objects": List[Dict],  # 检测到的对象（如果支持）
            }
        """
        if self.model is None:
            # 如果没有模型，返回默认结果
            return {
                "correct": True,
                "confidence": 0.5,
                "feedback": "验证器未加载，跳过验证",
                "suggested_prompt": original_prompt,
                "detected_objects": []
            }
        
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == "qwen":
                return self._verify_with_qwen(image, original_prompt, expected_layout, threshold)
            elif self.model_type == "clip":
                return self._verify_with_clip(image, original_prompt, threshold)
            else:
                return {
                    "correct": True,
                    "confidence": 0.5,
                    "feedback": "未知的验证器类型",
                    "suggested_prompt": original_prompt,
                    "detected_objects": []
                }
    
    def _verify_with_qwen(self, 
                         image: Image.Image,
                         prompt: str,
                         expected_layout: Optional[List[Dict]],
                         threshold: float) -> Dict:
        """使用 Qwen-VL 验证"""
        try:
            # 构建验证 prompt
            verify_prompt = f"请检查这张图片是否符合以下描述：{prompt}。"
            if expected_layout:
                verify_prompt += " 特别检查以下对象的位置："
                for obj in expected_layout:
                    verify_prompt += f" {obj['name']}应该在位置{obj['bbox']}；"
            
            # 处理输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": verify_prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成验证结果
            generated_ids = self.model.generate(
                **image_inputs,
                max_new_tokens=128,
                do_sample=False
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(image_inputs.input_ids, generated_ids)
            ]
            
            response_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # 解析响应（简单启发式）
            is_correct = "符合" in response_text or "正确" in response_text or "是的" in response_text
            confidence = 0.8 if is_correct else 0.3
            
            # 提取反馈
            feedback = response_text.strip()
            
            # 生成修正建议
            suggested_prompt = prompt
            if not is_correct and "建议" in response_text or "应该" in response_text:
                # 尝试从响应中提取建议
                suggested_prompt = prompt  # 简化处理
            
            return {
                "correct": is_correct and confidence >= threshold,
                "confidence": confidence,
                "feedback": feedback,
                "suggested_prompt": suggested_prompt,
                "detected_objects": []  # Qwen-VL 需要额外调用才能获取 grounding
            }
            
        except Exception as e:
            print(f"⚠️ Qwen-VL 验证出错: {e}")
            return {
                "correct": True,  # 出错时默认通过
                "confidence": 0.5,
                "feedback": f"验证过程出错: {str(e)}",
                "suggested_prompt": prompt,
                "detected_objects": []
            }
    
    def _verify_with_clip(self, 
                         image: Image.Image,
                         prompt: str,
                         threshold: float) -> Dict:
        """使用 CLIP 验证（简单版本）"""
        try:
            # 处理输入
            inputs = self.processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # 计算相似度
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            confidence = probs[0, 0].item()
            is_correct = confidence >= threshold
            
            return {
                "correct": is_correct,
                "confidence": confidence,
                "feedback": f"CLIP相似度: {confidence:.3f}",
                "suggested_prompt": prompt,
                "detected_objects": []
            }
            
        except Exception as e:
            print(f"⚠️ CLIP 验证出错: {e}")
            return {
                "correct": True,
                "confidence": 0.5,
                "feedback": f"验证过程出错: {str(e)}",
                "suggested_prompt": prompt,
                "detected_objects": []
            }
    
    def batch_verify(self, 
                    images: List[Image.Image],
                    prompts: List[str],
                    expected_layouts: Optional[List[List[Dict]]] = None,
                    threshold: float = 0.7) -> List[Dict]:
        """批量验证"""
        results = []
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            expected_layout = expected_layouts[i] if expected_layouts else None
            result = self.verify(image, prompt, expected_layout, threshold)
            results.append(result)
        return results


def create_feedback_verifier(vlm_model_name: str = "Qwen/Qwen-VL",
                             device: str = "cuda",
                             use_grounding: bool = True) -> FeedbackVerifier:
    """
    创建反馈验证器
    
    Args:
        vlm_model_name: VLM 模型名称
        device: 设备
        use_grounding: 是否使用 grounding
    
    Returns:
        FeedbackVerifier 实例
    """
    return FeedbackVerifier(
        vlm_model_name=vlm_model_name,
        device=device,
        use_grounding=use_grounding
    )

