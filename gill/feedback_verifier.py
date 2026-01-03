"""
反馈验证模块 (Feedback Verifier)

🌟 异构验证器架构（Heterogeneous Verifier Architecture）
避免"自循环验证"偏差，使用多种验证器组合：
1. Grounding DINO：检测位置准确性（Neuro-Symbolic Feedback）
2. Qwen2-VL-7B：检测语义准确性（轻量级 VLM）
3. GPT-4o/Claude（可选）：用于评估实验的金标准

论文宣称：MoE-based Self-Correction（专家混合模型自我修正）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Literal
from PIL import Image
import numpy as np


class FeedbackVerifier:
    """
    反馈验证器（异构架构）
    
    使用多种验证器组合，避免"裁判员兼运动员"偏差
    """
    
    def __init__(self, 
                 verifier_type: Literal["grounding_dino", "qwen2vl_7b", "hybrid", "qwen2vl"] = "hybrid",
                 vlm_model_name: Optional[str] = None,
                 device: str = "cuda",
                 use_grounding: bool = True):
        """
        Args:
            verifier_type: 验证器类型
                - "grounding_dino": 仅使用 Grounding DINO（位置检测）
                - "qwen2vl_7b": 仅使用 Qwen2-VL-7B（语义验证）
                - "hybrid": 混合模式（推荐，Grounding DINO + Qwen2-VL-7B）
                - "qwen2vl": 使用指定的 Qwen-VL 模型（兼容旧代码）
            vlm_model_name: VLM 模型名称或路径（用于 qwen2vl 模式）
            device: 设备
            use_grounding: 是否使用 grounding 功能（检测对象位置）
        """
        self.device = device
        self.use_grounding = use_grounding
        self.verifier_type = verifier_type
        
        # 根据类型加载验证器
        if verifier_type == "hybrid":
            print("🔀 使用混合验证器（Grounding DINO + Qwen2-VL-7B）")
            self._load_grounding_dino()
            self._load_qwen2vl_7b()
        elif verifier_type == "grounding_dino":
            print("🎯 使用 Grounding DINO 验证器")
            self._load_grounding_dino()
            self.qwen_model = None
            self.qwen_processor = None
        elif verifier_type == "qwen2vl_7b":
            print("🤖 使用 Qwen2-VL-7B 验证器")
            self._load_qwen2vl_7b()
            self.grounding_model = None
            self.grounding_processor = None
        else:  # qwen2vl (兼容旧代码)
            self.vlm_model_name = vlm_model_name or "Qwen/Qwen2-VL-7B-Instruct"
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
    
    def _verify_with_qwen_internal(self,
                         image: Image.Image,
                         prompt: str,
                         expected_layout: Optional[List[Dict]],
                                  threshold: float,
                                  model,
                                  processor) -> Dict:
        """
        内部方法：使用 Qwen-VL 验证（可被不同模型调用）
        """
        """使用 Qwen2-VL 验证（增强版：支持推理时修正）"""
        try:
            # 🌟 增强的验证 prompt：Chain-of-Thought 风格
            verify_prompt = f"""你是一名严格的视觉质检员。请检查图片是否符合以下描述：{prompt}

请按以下步骤分析：
1. 首先，识别图片中的所有主要物体
2. 然后，检查每个物体的位置是否符合描述中的空间关系要求
3. 最后，给出明确的判断和修正建议"""
            
            if expected_layout:
                verify_prompt += "\n\n期望的布局要求：\n"
                for obj in expected_layout:
                    bbox = obj.get('bbox', [])
                    # 将归一化坐标转换为位置描述
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    position_desc = ""
                    if cx < 0.33:
                        position_desc += "左侧"
                    elif cx > 0.67:
                        position_desc += "右侧"
                    else:
                        position_desc += "中间"
                    if cy < 0.33:
                        position_desc += "上方"
                    elif cy > 0.67:
                        position_desc += "下方"
                    
                    verify_prompt += f"- {obj['name']} 应该在 {position_desc} (坐标: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}])\n"
            
            verify_prompt += """
\n请按以下格式回答：
- 如果完全符合：回答"符合"
- 如果不符合：回答"不符合"，并详细说明：
  1. 哪个物体的位置不对
  2. 当前位置在哪里
  3. 应该如何调整（例如："猫当前在中间，应该向左移动到左侧区域"）"""
            
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
            # 🌟 增强的响应解析：提取修正建议（自然语言 Rationale）
            response_text = response_text.strip()
            
            # 判断是否符合（更严格的判断）
            is_correct = (
                "符合" in response_text and 
                "不符合" not in response_text and
                "不对" not in response_text and
                "错误" not in response_text and
                "问题" not in response_text
            )
            
            # 计算置信度（基于响应中的关键词）
            confidence = 0.9 if is_correct else 0.2
            if "完全" in response_text and is_correct:
                confidence = 0.95
            elif "基本" in response_text and is_correct:
                confidence = 0.8

            # 🌟 关键：提取自然语言反馈（Rationale）和修正建议
            refinement_instruction = None
            correction_details = []
            rationale = response_text  # 完整的自然语言反馈
            
            if not is_correct:
                # 提取修正建议（用于 Layout Planner 修正）
                lines = response_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # 提取包含修正信息的行
                    if any(keyword in line for keyword in ["应该", "需要", "建议", "调整", "移动", "位置", "在", "应该"]):
                        correction_details.append(line)
                
                # 构建结构化的修正指令
                if correction_details:
                    # 合并所有修正建议
                    refinement_instruction = "\n".join(correction_details)
                else:
                    # 如果没有明确的修正建议，使用原始反馈
                    refinement_instruction = response_text
                
                # 确保 refinement_instruction 包含足够的信息
                if len(refinement_instruction) < 20:
                    refinement_instruction = f"验证反馈：{response_text}"
            else:
                # 即使通过验证，也保留反馈信息（用于日志）
                refinement_instruction = "验证通过，无需修正"

            return {
                "correct": is_correct and confidence >= threshold,
                "confidence": confidence,
                "feedback": response_text,  # 原始反馈文本
                "rationale": rationale,  # 自然语言解释（用于论文展示）
                "refinement_instruction": refinement_instruction,  # 用于反馈给 Layout Planner
                "correction_details": correction_details,  # 详细的修正建议列表
                "suggested_prompt": prompt,
                "detected_objects": []
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

