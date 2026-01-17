#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理代理脚本 (Inference Agent)
实现完整的"生成-验证-修改"闭环链路

核心流程：
1. User Prompt → Layout Planner (生成布局)
2. Layout → Spatial Adapter → Image Generation (生成图像)
3. Image → Feedback Verifier (验证)
4. 如果失败 → 反馈给 Layout Planner → 重新生成
5. 重复直到成功或达到最大重试次数

这是将项目从"工程复现"提升到"算法创新"的关键模块。
"""

import argparse
import os
import torch
from PIL import Image
from typing import Dict, List, Optional
import json
from pathlib import Path

# 导入项目模块
from gill.models import GILL, GILLArgs
from gill.layout_planner import LayoutPlanner, create_layout_planner_from_gill
from gill.feedback_verifier import FeedbackVerifier, create_feedback_verifier
from gill.spatial_adapter import create_spatial_adapter_for_kolors, load_spatial_adapter_state_dict


class InferenceAgent:
    """
    推理代理：实现生成-验证-修改闭环
    """
    
    def __init__(
        self,
        gill_model_path: Optional[str] = None,
        layout_planner_path: Optional[str] = None,
        spatial_adapter_path: Optional[str] = None,
        verifier_model_path: str = "/mnt/disk/lxh/models/Qwen2.5-VL-7B-Instruct",
        verifier_type: str = "hybrid",  # 添加 verifier_type 参数
        device: str = "cuda",
        max_retries: int = 3,
        enable_cot: bool = True  # Chain-of-Thought
    ):
        """
        Args:
            gill_model_path: GILL 模型路径（可选，如果使用 Kolors 可直接用 Kolors）
            layout_planner_path: Layout Planner 模型路径
            spatial_adapter_path: Spatial Adapter 模型路径
            verifier_model_path: 验证器模型路径（Qwen2-VL）
            device: 设备
            max_retries: 最大重试次数
            enable_cot: 是否启用 Chain-of-Thought（思考过程）
        """
        self.device = device
        self.max_retries = max_retries
        self.enable_cot = enable_cot
        self.verifier_type = verifier_type  # 保存 verifier_type
        
        print("🚀 初始化推理代理...")
        
        # 1. 加载 GILL/Kolors 模型
        self._load_gill_model(gill_model_path)
        
        # 2. 加载 Layout Planner
        self._load_layout_planner(layout_planner_path)
        
        # 3. 加载 Spatial Adapter
        self._load_spatial_adapter(spatial_adapter_path)
        
        # 4. 加载 Feedback Verifier（异构验证器）
        self._load_verifier(verifier_model_path, verifier_type=verifier_type)
        
        print("✅ 推理代理初始化完成！")
    
    def _load_gill_model(self, model_path: Optional[str]):
        """加载 GILL/Kolors 模型"""
        print("📦 加载 GILL/Kolors 模型...")
        try:
            # 如果提供了模型路径，加载 GILL
            # 否则使用 Kolors（通过 GILL 的 is_kolors 模式）
            model_args = GILLArgs()
            self.gill_model = GILL(
                tokenizer=None,  # Kolors 模式下不需要
                model_args=model_args,
                load_sd=True,  # 加载 SDXL/Kolors
                device_map=self.device
            )
            print("  ✅ GILL/Kolors 模型加载完成")
        except Exception as e:
            print(f"  ⚠️ GILL 模型加载失败: {e}")
            print("  ℹ️ 将尝试使用 Kolors 直接生成")
            self.gill_model = None
    
    def _load_layout_planner(self, model_path: Optional[str]):
        """加载 Layout Planner"""
        print("📦 加载 Layout Planner...")
        if model_path and os.path.exists(model_path):
            try:
                adapter_config = os.path.join(model_path, "adapter_config.json")
                if os.path.isdir(model_path) and os.path.exists(adapter_config):
                    # LoRA/PEFT 适配器
                    try:
                        from peft import PeftConfig, PeftModel
                        peft_config = PeftConfig.from_pretrained(model_path)
                        base_model_path = peft_config.base_model_name_or_path
                        self.layout_planner = LayoutPlanner(
                            base_model_path,
                            device=self.device,
                            use_lora=False
                        )
                        self.layout_planner.model = PeftModel.from_pretrained(
                            self.layout_planner.model,
                            model_path
                        )
                        self.layout_planner.model.eval()
                        print("  ✅ Layout Planner (LoRA) 加载完成")
                    except Exception as e:
                        print(f"  ⚠️ LoRA 适配器加载失败: {e}")
                        self.layout_planner = None
                else:
                    # 完整模型目录或单一模型路径
                    self.layout_planner = LayoutPlanner(
                        model_path,
                        device=self.device,
                        use_lora=False
                    )
                    print("  ✅ Layout Planner 加载完成")
            except Exception as e:
                print(f"  ⚠️ Layout Planner 加载失败: {e}")
                self.layout_planner = None
        else:
            # 如果没有提供路径，创建一个基础的 Layout Planner
            # 实际使用时应该加载训练好的模型
            print("  ⚠️ 未提供 Layout Planner 路径，将使用基础版本")
            self.layout_planner = None
    
    def _load_spatial_adapter(self, model_path: Optional[str]):
        """加载 Spatial Adapter"""
        print("📦 加载 Spatial Adapter...")
        self.spatial_adapter = create_spatial_adapter_for_kolors()
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.spatial_adapter = load_spatial_adapter_state_dict(
                    state_dict,
                    device=self.device,
                    dtype=torch.float32
                )
                print(f"  ✅ Spatial Adapter 权重已载入 (模块数: {len(self.spatial_adapter)})")
            except Exception as e:
                print(f"  ⚠️ Spatial Adapter 加载失败: {e}")
        else:
            print("  ⚠️ 未提供 Spatial Adapter 路径，将使用默认版本")
    
    def _load_verifier(self, model_path: str, verifier_type: str = "hybrid"):
        """
        加载 Feedback Verifier（异构验证器）
        
        Args:
            model_path: 模型路径（用于兼容旧代码）
            verifier_type: 验证器类型
                - "hybrid": 混合模式（Grounding DINO + Qwen2-VL-7B，推荐）
                - "grounding_dino": 仅使用 Grounding DINO
                - "qwen2vl_7b": 仅使用 Qwen2-VL-7B
        """
        print(f"📦 加载 Feedback Verifier (类型: {verifier_type})...")
        try:
            if verifier_type == "hybrid":
                # 🌟 推荐：混合验证器（避免自循环验证偏差）
                self.verifier = create_feedback_verifier(
                    verifier_type="hybrid",
                    device=self.device,
                    use_grounding=True
                )
            elif verifier_type == "qwen2vl_7b":
                # 仅使用 Qwen2-VL-7B（轻量级）
                self.verifier = create_feedback_verifier(
                    verifier_type="qwen2vl_7b",
                    device=self.device,
                    use_grounding=True
                )
            else:
                # 兼容旧代码
                self.verifier = create_feedback_verifier(
                    verifier_type="qwen2vl",
                    vlm_model_name=model_path,
                    device=self.device,
                    use_grounding=True
                )
            print("  ✅ Feedback Verifier 加载完成")
        except Exception as e:
            print(f"  ⚠️ Feedback Verifier 加载失败: {e}")
            self.verifier = None
    
    def generate_with_feedback_loop(
        self,
        prompt: str,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        save_intermediate: bool = True,
        output_dir: str = "./outputs"
    ) -> Dict:
        """
        执行生成-验证-修改闭环
        
        Args:
            prompt: 用户输入的文本提示
            guidance_scale: 引导强度
            num_inference_steps: 推理步数
            save_intermediate: 是否保存中间结果
            output_dir: 输出目录
        
        Returns:
            {
                "final_image": Image,
                "layout_history": List[Dict],  # 布局历史
                "feedback_history": List[Dict],  # 反馈历史
                "success": bool,
                "num_attempts": int
            }
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🎨 开始生成: {prompt}")
        print(f"{'='*60}\n")
        
        layout_history = []
        feedback_history = []
        image_history = []
        
        for attempt in range(self.max_retries + 1):
            print(f"\n🔄 尝试 {attempt + 1}/{self.max_retries + 1}")
            print("-" * 60)
            
            # Step 1: Layout Planning（带 CoT）
            print("📐 Step 1: 布局规划...")
            layout_result = self._plan_layout(prompt, attempt, feedback_history)
            layout_history.append(layout_result)
            
            if layout_result.get("objects") is None or len(layout_result["objects"]) == 0:
                print("  ⚠️ 布局规划失败，跳过本次尝试")
                continue
            
            print(f"  ✅ 规划了 {len(layout_result['objects'])} 个对象")
            for obj in layout_result["objects"]:
                print(f"     - {obj['name']}: {obj['bbox']}")
            
            # Step 2: Image Generation
            print("\n🎨 Step 2: 图像生成...")
            generated_image = self._generate_image(
                prompt=prompt,
                layout=layout_result,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            if generated_image is None:
                print("  ⚠️ 图像生成失败，跳过本次尝试")
                continue
            
            image_history.append(generated_image)
            
            # 保存中间结果
            if save_intermediate:
                intermediate_path = os.path.join(
                    output_dir,
                    f"attempt_{attempt + 1}_layout_{len(layout_result['objects'])}_objects.png"
                )
                generated_image.save(intermediate_path)
                print(f"  💾 已保存: {intermediate_path}")
            
            # Step 3: Verification
            print("\n🔍 Step 3: 验证生成结果...")
            feedback = self._verify_image(
                image=generated_image,
                prompt=prompt,
                expected_layout=layout_result.get("objects")
            )
            feedback_history.append(feedback)
            
            print(f"  {'✅' if feedback.get('correct') else '❌'} 验证结果: {feedback.get('feedback', '')[:100]}")
            
            # Step 4: 判断是否需要重试
            if feedback.get("correct", False):
                print(f"\n🎉 生成成功！共尝试 {attempt + 1} 次")
                return {
                    "final_image": generated_image,
                    "layout_history": layout_history,
                    "feedback_history": feedback_history,
                    "image_history": image_history,
                    "success": True,
                    "num_attempts": attempt + 1,
                    "final_layout": layout_result
                }
            else:
                if attempt < self.max_retries:
                    refinement = feedback.get("refinement_instruction", "")
                    print(f"\n  💡 修正建议: {refinement[:200]}")
                    print(f"  🔄 将根据反馈调整布局并重试...")
                else:
                    print(f"\n⚠️ 达到最大重试次数 ({self.max_retries + 1})，停止生成")
        
        # 所有尝试都失败
        return {
            "final_image": image_history[-1] if image_history else None,
            "layout_history": layout_history,
            "feedback_history": feedback_history,
            "image_history": image_history,
            "success": False,
            "num_attempts": self.max_retries + 1,
            "final_layout": layout_history[-1] if layout_history else None
        }
    
    def _plan_layout(
        self,
        prompt: str,
        attempt: int,
        feedback_history: List[Dict]
    ) -> Dict:
        """
        布局规划（支持反馈修正）
        
        这是闭环的核心：根据验证反馈修正布局
        """
        if self.layout_planner is None:
            # 如果没有 Layout Planner，返回空布局
            return {"objects": [], "layout_text": ""}
        
        # 🌟 关键逻辑：构建带反馈的 prompt
        current_prompt = prompt
        feedback_text = None
        
        if attempt > 0 and feedback_history:
            last_feedback = feedback_history[-1]
            
            # 优先使用 refinement_instruction（结构化反馈）
            refinement = last_feedback.get("refinement_instruction", "")
            feedback_raw = last_feedback.get("feedback", "")
            
            if refinement:
                # 使用结构化的修正建议
                feedback_text = refinement
                current_prompt = f"""{prompt}

上一轮生成结果存在问题：
{refinement}

请根据以上反馈重新规划布局，确保修正这些错误。"""
            elif feedback_raw and "不符合" in feedback_raw:
                # 如果没有结构化反馈，使用原始反馈文本
                feedback_text = feedback_raw
                current_prompt = f"""{prompt}

上一轮验证反馈：
{feedback_raw}

请分析反馈中的问题，并重新规划布局以修正错误。"""
        
        # 🌟 Chain-of-Thought: 如果启用，让模型先"思考"再输出布局
        enable_cot = self.enable_cot
        
        # 生成布局（传递 feedback 参数）
        layout_result = self.layout_planner.generate_layout(
            current_prompt,
            apply_refinement=True,
            enable_cot=enable_cot,
            feedback=feedback_text  # 传递反馈给 Layout Planner
        )
        
        return layout_result
    
    def _generate_image(
        self,
        prompt: str,
        layout: Dict,
        guidance_scale: float,
        num_inference_steps: int
    ) -> Optional[Image.Image]:
        """生成图像"""
        if self.gill_model is None:
            print("  ⚠️ GILL 模型未加载，无法生成图像")
            return None
        
        try:
            # 使用 GILL 的 generate_with_layout 方法
            result = self.gill_model.generate_with_layout(
                prompt=prompt,
                enable_layout=True,
                enable_feedback=False,  # 在 agent 层面处理反馈
                layout_planner=None,  # 已经规划好了
                spatial_adapter=self.spatial_adapter,
                feedback_verifier=None,  # 在 agent 层面验证
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_retries=1  # 只生成一次，重试在 agent 层面
            )
            
            return result.get("image")
        except Exception as e:
            print(f"  ⚠️ 图像生成出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _verify_image(
        self,
        image: Image.Image,
        prompt: str,
        expected_layout: Optional[List[Dict]]
    ) -> Dict:
        """验证图像"""
        if self.verifier is None:
            # 如果没有验证器，默认通过
            return {
                "correct": True,
                "confidence": 0.5,
                "feedback": "验证器未加载，跳过验证"
            }
        
        return self.verifier.verify(
            image=image,
            original_prompt=prompt,
            expected_layout=expected_layout,
            threshold=0.7
        )


def main():
    parser = argparse.ArgumentParser(
        description="推理代理：实现生成-验证-修改闭环"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="输入提示词"
    )
    parser.add_argument(
        "--layout_planner_path",
        type=str,
        default=None,
        help="Layout Planner 模型路径"
    )
    parser.add_argument(
        "--spatial_adapter_path",
        type=str,
        default=None,
        help="Spatial Adapter 模型路径"
    )
    parser.add_argument(
        "--verifier_model_path",
        type=str,
        default="/mnt/disk/lxh/models/Qwen2.5-VL-7B-Instruct",
        help="验证器模型路径（Qwen2.5-VL-7B-Instruct）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="最大重试次数"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="引导强度"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--enable_cot",
        action="store_true",
        help="启用 Chain-of-Thought（思考过程）"
    )
    
    args = parser.parse_args()
    
    # 创建推理代理
    agent = InferenceAgent(
        layout_planner_path=args.layout_planner_path,
        spatial_adapter_path=args.spatial_adapter_path,
        verifier_model_path=args.verifier_model_path,
        device=args.device,
        max_retries=args.max_retries,
        enable_cot=args.enable_cot
    )
    
    # 执行生成
    result = agent.generate_with_feedback_loop(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        save_intermediate=True,
        output_dir=args.output_dir
    )
    
    # 保存最终结果
    if result["success"]:
        final_path = os.path.join(args.output_dir, "final_result.png")
        result["final_image"].save(final_path)
        print(f"\n✅ 最终结果已保存: {final_path}")
    else:
        print(f"\n⚠️ 生成未完全成功，但已保存最后一次尝试的结果")
    
    # 保存历史记录
    history_path = os.path.join(args.output_dir, "generation_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "prompt": args.prompt,
            "success": result["success"],
            "num_attempts": result["num_attempts"],
            "layout_history": [
                {
                    "objects": layout.get("objects", []),
                    "layout_text": layout.get("layout_text", "")
                }
                for layout in result["layout_history"]
            ],
            "feedback_history": result["feedback_history"]
        }, f, ensure_ascii=False, indent=2)
    print(f"📝 生成历史已保存: {history_path}")


if __name__ == "__main__":
    main()
