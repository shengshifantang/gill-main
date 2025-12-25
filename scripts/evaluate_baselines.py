#!/usr/bin/env python3
"""
多层次 Baseline 对比评估框架

支持三类对比：
1. 消融实验 (Ablation): Base GILL, Heuristic Layout
2. 同类竞品 (SOTA): GLIGEN, ControlNet, Emu2
3. 通用模型 (Generalist): DALL-E 3, Midjourney

Usage:
    python scripts/evaluate_baselines.py \
        --test-set data/test_set.jsonl \
        --output-dir evaluation_results/baseline_comparison \
        --baselines base_gill heuristic ours \
        --metrics layout_iou clip_score fid
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.models import GILL, load_gill
from gill.layout_planner import LayoutPlanner


class BaselineGenerator:
    """统一的 Baseline 生成接口"""
    
    def __init__(self, baseline_type: str, config: Dict):
        self.baseline_type = baseline_type
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """根据 baseline 类型加载模型"""
        if self.baseline_type == "base_gill":
            # Baseline A: Base GILL (无 Layout Planner, 无 Spatial Adapter)
            self.model = load_gill(
                gill_model=self.config.get("gill_model", "./checkpoints/gill_opt"),
                load_sd=True,
                device=self.config.get("device", "cuda:0")
            )
            # 确保不使用 Layout Planner 和 Spatial Adapter
            self.model.use_layout_planner = False
        
        elif self.baseline_type == "heuristic":
            # Baseline B: Heuristic Layout (有 Spatial Adapter, 但用规则生成坐标)
            self.model = load_gill(
                gill_model=self.config.get("gill_model", "./checkpoints/gill_opt"),
                load_sd=True,
                device=self.config.get("device", "cuda:0")
            )
            # 使用启发式布局生成器（不使用 LLM Planner）
            from gill.layout_planner import heuristic_layout_from_caption
            self.heuristic_fn = heuristic_layout_from_caption
        
        elif self.baseline_type == "ours":
            # Ours: Full Pipeline (CoT Layout Planner + Spatial Adapter)
            self.model = load_gill(
                gill_model=self.config.get("gill_model", "./checkpoints/gill_opt"),
                load_sd=True,
                device=self.config.get("device", "cuda:0")
            )
            # 加载 Layout Planner
            planner_model = self.config.get("planner_model", "./checkpoints/layout_planner_cot_15k")
            self.planner = LayoutPlanner(planner_model, device=self.config.get("device", "cuda:0"))
        
        elif self.baseline_type == "gligen":
            # Baseline C: GLIGEN (需要手动输入 bbox)
            # 这里可以加载 GLIGEN 模型或使用类似的实现
            raise NotImplementedError("GLIGEN baseline 需要单独实现")
        
        elif self.baseline_type == "dalle3":
            # Baseline E: DALL-E 3 (通过 API)
            # 需要 OpenAI API Key
            self.api_key = self.config.get("openai_api_key")
            if not self.api_key:
                raise ValueError("DALL-E 3 需要 OpenAI API Key")
        
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")
    
    def generate(self, prompt: str, **kwargs) -> Tuple[Optional[torch.Tensor], Optional[Dict]]:
        """
        生成图像
        
        Returns:
            (image_tensor, metadata)
            - image_tensor: 生成的图像 (PIL Image 或 torch.Tensor)
            - metadata: 包含布局信息、推理过程等
        """
        if self.baseline_type == "base_gill":
            # 直接使用 prompt 生成，无布局控制
            result = self.model.generate_for_images_and_texts(
                [prompt],
                num_words=16,
                guidance_scale=7.5
            )
            image = result[0] if result else None
            metadata = {"layout_used": False, "bboxes": None}
            return image, metadata
        
        elif self.baseline_type == "heuristic":
            # 使用启发式规则生成布局
            from gill.layout_planner import heuristic_layout_from_caption
            objects = heuristic_layout_from_caption(prompt)
            
            if not objects:
                # 如果启发式失败，回退到 base_gill
                return self._generate_base(prompt)
            
            # 转换为 bbox 格式
            bboxes = torch.tensor([[obj["bbox"][0], obj["bbox"][1], 
                                   obj["bbox"][2], obj["bbox"][3]] 
                                  for obj in objects], dtype=torch.float32)
            
            # 使用 Spatial Adapter 生成
            result = self.model.generate_with_layout(
                prompt=prompt,
                objects=[obj["name"] for obj in objects],
                bboxes=bboxes.unsqueeze(0),
                enable_layout=True,
                spatial_adapter=self.model.spatial_adapter if hasattr(self.model, 'spatial_adapter') else None
            )
            
            image = result.get("generated_image") if isinstance(result, dict) else result
            metadata = {
                "layout_used": True,
                "layout_method": "heuristic",
                "bboxes": bboxes.tolist(),
                "objects": objects
            }
            return image, metadata
        
        elif self.baseline_type == "ours":
            # 使用 CoT Layout Planner + Spatial Adapter
            # 1. 使用 Planner 生成布局
            layout_output = self.planner.plan_layout(prompt)
            objects, bboxes = self.planner.parse_layout_output(layout_output)
            
            if not objects or bboxes is None:
                return None, {"error": "Layout planning failed"}
            
            # 2. 使用 Spatial Adapter 生成
            result = self.model.generate_with_layout(
                prompt=prompt,
                objects=objects,
                bboxes=bboxes,
                enable_layout=True,
                spatial_adapter=self.model.spatial_adapter if hasattr(self.model, 'spatial_adapter') else None
            )
            
            image = result.get("generated_image") if isinstance(result, dict) else result
            metadata = {
                "layout_used": True,
                "layout_method": "cot_planner",
                "bboxes": bboxes.tolist() if isinstance(bboxes, torch.Tensor) else bboxes,
                "objects": objects,
                "cot_reasoning": layout_output.get("reasoning", "") if isinstance(layout_output, dict) else ""
            }
            return image, metadata
        
        elif self.baseline_type == "dalle3":
            # 通过 OpenAI API 调用 DALL-E 3
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                
                image_url = response.data[0].url
                # 下载图片
                import requests
                from PIL import Image
                img_response = requests.get(image_url)
                image = Image.open(io.BytesIO(img_response.content))
                
                metadata = {
                    "layout_used": False,
                    "model": "dall-e-3",
                    "api_call": True
                }
                return image, metadata
            except Exception as e:
                return None, {"error": str(e)}
        
        return None, {"error": "Unknown baseline type"}
    
    def _generate_base(self, prompt: str):
        """Base GILL 生成（无布局控制）"""
        result = self.model.generate_for_images_and_texts(
            [prompt],
            num_words=16,
            guidance_scale=7.5
        )
        image = result[0] if result else None
        metadata = {"layout_used": False}
        return image, metadata


class LayoutEvaluator:
    """布局准确率评估器（使用 GroundingDINO 或 YOLO-World）"""
    
    def __init__(self, detector_type: str = "grounding_dino"):
        self.detector_type = detector_type
        self.detector = None
        self._load_detector()
    
    def _load_detector(self):
        """加载目标检测器"""
        if self.detector_type == "grounding_dino":
            try:
                from groundingdino.util.inference import load_model, load_image, predict
                # 需要下载 GroundingDINO 模型
                self.detector = {
                    "model": load_model("groundingdino_swinb_cogcoor.pth", "groundingdino/config/GroundingDINO_SwinB.cfg.py"),
                    "predict_fn": predict
                }
            except ImportError:
                print("⚠️ GroundingDINO 未安装，将使用简化版评估")
                self.detector = None
        elif self.detector_type == "yolo_world":
            try:
                from ultralytics import YOLO
                self.detector = YOLO("yolov8x-world.pt")
            except ImportError:
                print("⚠️ YOLO-World 未安装，将使用简化版评估")
                self.detector = None
    
    def compute_layout_iou(self, image, prompt: str, predicted_bboxes: List[List[float]], 
                          object_names: List[str]) -> Dict:
        """
        计算布局 IoU
        
        Args:
            image: PIL Image 或 torch.Tensor
            prompt: 原始 prompt
            predicted_bboxes: 预测的 bbox 列表 [[x1,y1,x2,y2], ...]
            object_names: 物体名称列表
        
        Returns:
            {
                "mean_iou": float,
                "per_object_iou": List[float],
                "object_recall": float,
                "count_accuracy": float
            }
        """
        if self.detector is None:
            # 简化版：只返回占位符
            return {
                "mean_iou": 0.0,
                "per_object_iou": [0.0] * len(object_names),
                "object_recall": 0.0,
                "count_accuracy": 0.0
            }
        
        # 使用检测器检测图像中的物体
        detected_objects = self._detect_objects(image, prompt, object_names)
        
        # 计算 IoU
        ious = []
        for pred_bbox, obj_name in zip(predicted_bboxes, object_names):
            if obj_name in detected_objects:
                detected_bbox = detected_objects[obj_name]["bbox"]
                iou = self._compute_iou(pred_bbox, detected_bbox)
                ious.append(iou)
            else:
                ious.append(0.0)
        
        # 计算召回率
        detected_count = len(detected_objects)
        expected_count = len(object_names)
        object_recall = detected_count / expected_count if expected_count > 0 else 0.0
        
        # 计算数量准确率
        count_accuracy = 1.0 if detected_count == expected_count else 0.0
        
        return {
            "mean_iou": np.mean(ious) if ious else 0.0,
            "per_object_iou": ious,
            "object_recall": object_recall,
            "count_accuracy": count_accuracy
        }
    
    def _detect_objects(self, image, prompt: str, object_names: List[str]) -> Dict:
        """使用检测器检测物体"""
        # 实现检测逻辑
        # 这里需要根据具体的检测器 API 实现
        return {}
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算两个 bbox 的 IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class CLIPScoreEvaluator:
    """CLIP Score 评估器（中文 CLIP）"""
    
    def __init__(self, clip_model_path: str = "./model/chinese_clip_ViT-L-14"):
        self.clip_model_path = clip_model_path
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """加载中文 CLIP 模型"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.model = CLIPModel.from_pretrained(self.clip_model_path)
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_path)
            self.model.eval()
        except Exception as e:
            print(f"⚠️ CLIP 模型加载失败: {e}")
            self.model = None
    
    def compute_clip_score(self, image, text: str) -> float:
        """计算 CLIP Score"""
        if self.model is None:
            return 0.0
        
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
        
        return score


def evaluate_baselines(test_set_path: str, output_dir: str, baselines: List[str], 
                      metrics: List[str], config: Dict):
    """评估多个 Baseline"""
    
    # 加载测试集
    test_samples = []
    with open(test_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    
    print(f"📊 加载测试集: {len(test_samples)} 条")
    
    # 初始化评估器
    layout_evaluator = LayoutEvaluator() if "layout_iou" in metrics else None
    clip_evaluator = CLIPScoreEvaluator() if "clip_score" in metrics else None
    
    # 评估每个 baseline
    results = {}
    
    for baseline_name in baselines:
        print(f"\n{'='*60}")
        print(f"🔍 评估 Baseline: {baseline_name}")
        print(f"{'='*60}")
        
        # 初始化生成器
        baseline_config = config.get(baseline_name, {})
        generator = BaselineGenerator(baseline_name, baseline_config)
        
        baseline_results = {
            "layout_iou": [],
            "object_recall": [],
            "count_accuracy": [],
            "clip_score": [],
            "metadata": []
        }
        
        for sample in tqdm(test_samples, desc=f"生成 {baseline_name}"):
            prompt = sample.get("caption", sample.get("prompt", ""))
            gt_objects = sample.get("objects", [])
            
            # 生成图像
            image, metadata = generator.generate(prompt)
            
            if image is None:
                continue
            
            # 保存图像
            os.makedirs(os.path.join(output_dir, baseline_name, "images"), exist_ok=True)
            image_path = os.path.join(output_dir, baseline_name, "images", 
                                     f"{sample.get('id', len(baseline_results['metadata']))}.png")
            if hasattr(image, 'save'):
                image.save(image_path)
            else:
                from PIL import Image
                if isinstance(image, torch.Tensor):
                    image = Image.fromarray(image.cpu().numpy())
                image.save(image_path)
            
            # 评估指标
            if "layout_iou" in metrics and layout_evaluator and metadata.get("bboxes"):
                layout_metrics = layout_evaluator.compute_layout_iou(
                    image, prompt, 
                    metadata.get("bboxes", []),
                    [obj.get("name", "") for obj in metadata.get("objects", [])]
                )
                baseline_results["layout_iou"].append(layout_metrics["mean_iou"])
                baseline_results["object_recall"].append(layout_metrics["object_recall"])
                baseline_results["count_accuracy"].append(layout_metrics["count_accuracy"])
            
            if "clip_score" in metrics and clip_evaluator:
                clip_score = clip_evaluator.compute_clip_score(image, prompt)
                baseline_results["clip_score"].append(clip_score)
            
            baseline_results["metadata"].append({
                "prompt": prompt,
                "image_path": image_path,
                "metadata": metadata
            })
        
        # 计算平均值
        results[baseline_name] = {
            "mean_layout_iou": np.mean(baseline_results["layout_iou"]) if baseline_results["layout_iou"] else 0.0,
            "mean_object_recall": np.mean(baseline_results["object_recall"]) if baseline_results["object_recall"] else 0.0,
            "mean_count_accuracy": np.mean(baseline_results["count_accuracy"]) if baseline_results["count_accuracy"] else 0.0,
            "mean_clip_score": np.mean(baseline_results["clip_score"]) if baseline_results["clip_score"] else 0.0,
            "samples": baseline_results["metadata"]
        }
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成对比表格
    generate_comparison_table(results, output_dir)
    
    print(f"\n✅ 评估完成！结果保存在: {output_dir}")


def generate_comparison_table(results: Dict, output_dir: str):
    """生成对比表格（LaTeX 和 CSV）"""
    
    # CSV 格式
    csv_lines = ["Method,Layout IoU,Object Recall,Count Accuracy,CLIP Score"]
    for baseline_name, metrics in results.items():
        csv_lines.append(
            f"{baseline_name},"
            f"{metrics['mean_layout_iou']:.4f},"
            f"{metrics['mean_object_recall']:.4f},"
            f"{metrics['mean_count_accuracy']:.4f},"
            f"{metrics['mean_clip_score']:.4f}"
        )
    
    with open(os.path.join(output_dir, "comparison_table.csv"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_lines))
    
    # LaTeX 格式
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Layout IoU $\\uparrow$ & Object Recall $\\uparrow$ & Count Acc. $\\uparrow$ & CLIP Score $\\uparrow$ \\\\",
        "\\midrule"
    ]
    
    for baseline_name, metrics in results.items():
        baseline_display = {
            "base_gill": "Base GILL",
            "heuristic": "Heuristic Layout",
            "ours": "Ours",
            "gligen": "GLIGEN",
            "dalle3": "DALL-E 3"
        }.get(baseline_name, baseline_name)
        
        latex_lines.append(
            f"{baseline_display} & "
            f"{metrics['mean_layout_iou']:.3f} & "
            f"{metrics['mean_object_recall']:.3f} & "
            f"{metrics['mean_count_accuracy']:.3f} & "
            f"{metrics['mean_clip_score']:.3f} \\\\"
        )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Baseline comparison results.}",
        "\\label{tab:baseline_comparison}",
        "\\end{table}"
    ])
    
    with open(os.path.join(output_dir, "comparison_table.tex"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✓ 对比表格已生成: {output_dir}/comparison_table.csv")


def main():
    parser = argparse.ArgumentParser(description="多层次 Baseline 对比评估")
    parser.add_argument("--test-set", type=str, required=True,
                       help="测试集 JSONL 文件")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--baselines", type=str, nargs='+',
                       default=["base_gill", "heuristic", "ours"],
                       choices=["base_gill", "heuristic", "ours", "gligen", "dalle3"],
                       help="要评估的 baseline 列表")
    parser.add_argument("--metrics", type=str, nargs='+',
                       default=["layout_iou", "clip_score"],
                       choices=["layout_iou", "clip_score", "fid"],
                       help="评估指标")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径（JSON）")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 设置默认配置
    for baseline in args.baselines:
        if baseline not in config:
            config[baseline] = {
                "device": args.device,
                "gill_model": "./checkpoints/gill_opt",
                "planner_model": "./checkpoints/layout_planner_cot_15k"
            }
    
    evaluate_baselines(
        args.test_set,
        args.output_dir,
        args.baselines,
        args.metrics,
        config
    )


if __name__ == "__main__":
    import io
    main()

