#!/usr/bin/env python3
"""
第一阶段训练验证脚本 - 修复版

修复内容：
1. ✅ 修复 negative_prompt 问题
2. ✅ 添加异常处理，避免单个样本失败导致整体中断
3. ✅ 添加图片有效性检查
4. ✅ 降低分辨率以节省显存（512x512）
"""

import sys
import os
import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import defaultdict
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.spatial_adapter_fixed import (
    load_spatial_adapter_state_dict,
    inject_spatial_control_to_unet,
    remove_spatial_control_from_unet,
)
from diffusers import KolorsPipeline


def analyze_gate_values(state_dict):
    """分析 Gate 参数的统计信息"""
    gate_values = []
    gate_info = []
    
    for name, param in state_dict.items():
        if "gate" in name.lower():
            val = param.detach().cpu().float()
            gate_values.append(val.flatten())
            gate_info.append({
                "name": name,
                "shape": list(param.shape),
                "mean": float(val.mean()),
                "std": float(val.std()),
                "min": float(val.min()),
                "max": float(val.max()),
                "tanh_mean": float(torch.tanh(val).mean()),
                "tanh_std": float(torch.tanh(val).std()),
            })
    
    if gate_values:
        all_gates = torch.cat(gate_values).numpy()
        all_gates_tanh = np.tanh(all_gates)
        
        return {
            "gate_info": gate_info,
            "statistics": {
                "total_gates": len(gate_values),
                "raw_mean": float(all_gates.mean()),
                "raw_std": float(all_gates.std()),
                "raw_min": float(all_gates.min()),
                "raw_max": float(all_gates.max()),
                "tanh_mean": float(all_gates_tanh.mean()),
                "tanh_std": float(all_gates_tanh.std()),
                "tanh_min": float(all_gates_tanh.min()),
                "tanh_max": float(all_gates_tanh.max()),
                "near_zero_ratio": float((np.abs(all_gates) < 0.01).sum() / len(all_gates)),
                "saturated_ratio": float((np.abs(all_gates_tanh) > 0.9).sum() / len(all_gates_tanh)),
            }
        }
    return None


def visualize_gate_distribution(gate_analysis, output_path):
    """可视化 Gate 分布"""
    if gate_analysis is None:
        return
    
    stats = gate_analysis["statistics"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Gate Parameter Analysis", fontsize=16, fontweight="bold")
    
    # 1. 统计信息文本
    ax = axes[0, 0]
    ax.axis("off")
    info_text = f"""
Gate Statistics:

Raw Values:
  Mean: {stats['raw_mean']:.4f}
  Std:  {stats['raw_std']:.4f}
  Range: [{stats['raw_min']:.4f}, {stats['raw_max']:.4f}]

After Tanh:
  Mean: {stats['tanh_mean']:.4f}
  Std:  {stats['tanh_std']:.4f}
  Range: [{stats['tanh_min']:.4f}, {stats['tanh_max']:.4f}]

Health Metrics:
  Near-zero ratio: {stats['near_zero_ratio']*100:.2f}%
  Saturated ratio: {stats['saturated_ratio']*100:.2f}%
  Total gates: {stats['total_gates']}
"""
    ax.text(0.1, 0.5, info_text, fontsize=11, family="monospace", va="center")
    
    # 2. 各层 Gate 值对比
    ax = axes[0, 1]
    gate_info = gate_analysis["gate_info"]
    layer_names = [g["name"].split(".")[-2] if "." in g["name"] else g["name"] for g in gate_info]
    gate_means = [g["tanh_mean"] for g in gate_info]
    
    ax.barh(range(len(layer_names)), gate_means, color="steelblue")
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_xlabel("Tanh(Gate) Mean")
    ax.set_title("Gate Values by Layer")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.3)
    
    # 3. Gate 原始值分布
    ax = axes[1, 0]
    raw_values = [g["mean"] for g in gate_info]
    ax.hist(raw_values, bins=20, color="coral", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Raw Gate Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Raw Gate Distribution")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Initial=0")
    ax.legend()
    
    # 4. 健康度评估
    ax = axes[1, 1]
    categories = ["Near Zero\n(Not Learned)", "Normal\n(Healthy)", "Saturated\n(Overfitting)"]
    values = [
        stats['near_zero_ratio'] * 100,
        (1 - stats['near_zero_ratio'] - stats['saturated_ratio']) * 100,
        stats['saturated_ratio'] * 100,
    ]
    colors = ["red", "green", "orange"]
    
    ax.pie(values, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Gate Health Distribution")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Gate 分析图保存到: {output_path}")


def draw_bboxes_on_image(image, bboxes, labels=None, color=(255, 0, 0), width=3):
    """在图像上绘制 bbox"""
    draw = ImageDraw.Draw(image)
    W, H = image.size
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        # 转换为像素坐标
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        
        # 绘制矩形
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # 绘制标签
        if labels and i < len(labels):
            label = labels[i]
            draw.text((x1 + 5, y1 + 5), label, fill=color)
    
    return image


def check_image_valid(image):
    """检查图片是否有效（非全黑）"""
    img_array = np.array(image)
    mean_val = img_array.mean()
    return mean_val > 1.0  # 如果均值小于1，说明几乎全黑


def generate_comparison(pipeline, caption, bboxes, obj_names, device, adapter_container=None, resolution=512):
    """生成对比图：无 Adapter vs 有 Adapter"""
    results = {}
    
    # 1. 无 Adapter 生成（Baseline）
    print("  生成 Baseline（无 Adapter）...")
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            image_baseline = pipeline(
                prompt=caption,
                negative_prompt="低质量，模糊，变形，丑陋",
                num_inference_steps=30,
                guidance_scale=5.0,
                height=resolution,
                width=resolution,
            ).images[0]
        
        if check_image_valid(image_baseline):
            results["baseline"] = image_baseline
            print("    ✓ Baseline 生成成功")
        else:
            print("    ⚠️  Baseline 图片异常（几乎全黑）")
            results["baseline"] = image_baseline  # 仍然保存，用于调试
            
    except Exception as e:
        print(f"    ❌ Baseline 生成失败: {e}")
        return results
    
    # 2. 有 Adapter 生成
    if adapter_container is not None:
        print("  生成 Adapter 控制图...")
        try:
            bboxes_tensor = torch.tensor([bboxes], device=device, dtype=torch.float32)
            
            orig_procs, _, _ = inject_spatial_control_to_unet(
                pipeline.unet,
                adapter_dict=adapter_container,
                bboxes=bboxes_tensor,
                phrase_embeddings=None,
                masks=None,
                adapter_dtype=torch.float16,
            )
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                image_adapter = pipeline(
                    prompt=caption,
                    negative_prompt="低质量，模糊，变形，丑陋",
                    num_inference_steps=30,
                    guidance_scale=5.0,
                    height=resolution,
                    width=resolution,
                ).images[0]
            
            remove_spatial_control_from_unet(pipeline.unet, orig_procs)
            
            if check_image_valid(image_adapter):
                results["adapter"] = image_adapter
                print("    ✓ Adapter 生成成功")
            else:
                print("    ⚠️  Adapter 图片异常（几乎全黑）")
                results["adapter"] = image_adapter
            
            # 3. 绘制 bbox 标注
            image_annotated = image_adapter.copy()
            image_annotated = draw_bboxes_on_image(image_annotated, bboxes, obj_names)
            results["annotated"] = image_annotated
            
        except Exception as e:
            print(f"    ❌ Adapter 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def validate_checkpoint(
    checkpoint_path,
    output_dir,
    kolors_path="./model/Kolors",
    test_data_path="./data/coco2014_cn_val_clean.jsonl",
    num_samples=5,
    device="cuda:0",
    resolution=512,
):
    """完整验证流程"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("第一阶段训练验证（修复版）")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"输出目录: {output_dir}")
    print(f"分辨率: {resolution}x{resolution}")
    print()
    
    # ==================== 1. 加载 Checkpoint ====================
    print("🔧 加载 Checkpoint...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "adapter" in state_dict:
        state_dict = state_dict["adapter"]
    
    print(f"✓ Checkpoint 加载成功，包含 {len(state_dict)} 个参数")
    print()
    
    # ==================== 2. Gate 分析 ====================
    print("📊 分析 Gate 参数...")
    gate_analysis = analyze_gate_values(state_dict)
    
    if gate_analysis:
        stats = gate_analysis["statistics"]
        print(f"  总 Gate 数: {stats['total_gates']}")
        print(f"  Tanh(Gate) 均值: {stats['tanh_mean']:.4f}")
        print(f"  Tanh(Gate) 标准差: {stats['tanh_std']:.4f}")
        print(f"  接近零比例: {stats['near_zero_ratio']*100:.2f}%")
        print(f"  饱和比例: {stats['saturated_ratio']*100:.2f}%")
        
        # 健康度判断
        if stats['near_zero_ratio'] > 0.8:
            print("  ⚠️  警告：超过 80% 的 Gate 接近零，可能未充分训练！")
        elif stats['saturated_ratio'] > 0.5:
            print("  ⚠️  警告：超过 50% 的 Gate 饱和，可能过拟合！")
        else:
            print("  ✅ Gate 参数健康")
        
        # 保存详细信息
        gate_json_path = output_dir / "gate_analysis.json"
        with open(gate_json_path, "w", encoding="utf-8") as f:
            json.dump(gate_analysis, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 详细分析保存到: {gate_json_path}")
        
        # 可视化
        visualize_gate_distribution(gate_analysis, output_dir / "gate_distribution.png")
    else:
        print("  ⚠️  未找到 Gate 参数")
    print()
    
    # ==================== 3. 生成效果测试 ====================
    print("🎨 测试生成效果...")
    print("🔧 加载 Kolors Pipeline...")
    
    pipeline = KolorsPipeline.from_pretrained(
        kolors_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    
    print("✓ Pipeline 加载成功")
    
    # 加载 Adapter
    print("🔧 加载 Adapter...")
    adapter_container = load_spatial_adapter_state_dict(
        state_dict,
        device=device,
        dtype=torch.float16
    )
    print("✓ Adapter 加载成功")
    print()
    
    # 加载测试数据
    print(f"🔧 加载测试数据: {test_data_path}")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f][:num_samples]
    print(f"✓ 加载 {len(test_data)} 个测试样本")
    print()
    
    # 生成对比图
    success_count = 0
    for i, sample in enumerate(test_data):
        try:
            caption = sample.get("caption", "")
            
            # 提取 bbox
            bboxes_list = []
            obj_names = []
            if "bboxes" in sample and sample["bboxes"]:
                bboxes_list = sample["bboxes"][:5]  # 限制最多5个bbox
                if "objects" in sample:
                    obj_names = [obj.get("name", "") for obj in sample["objects"]][:5]
            elif "objects" in sample and sample["objects"]:
                for obj in sample["objects"][:5]:
                    if "bbox" in obj:
                        bboxes_list.append(obj["bbox"])
                        obj_names.append(obj.get("name", ""))
            
            if not bboxes_list:
                print(f"[{i+1}/{len(test_data)}] 跳过（无 bbox）")
                continue
            
            print(f"[{i+1}/{len(test_data)}] {caption[:60]}...")
            print(f"  BBox 数量: {len(bboxes_list)}")
            
            # 生成对比
            results = generate_comparison(
                pipeline, caption, bboxes_list, obj_names, device, adapter_container, resolution
            )
            
            if not results:
                print(f"  ⚠️  生成失败，跳过")
                continue
            
            # 保存结果
            sample_dir = output_dir / f"sample_{i:02d}"
            sample_dir.mkdir(exist_ok=True)
            
            # 保存元数据
            meta = {
                "caption": caption,
                "bboxes": bboxes_list,
                "obj_names": obj_names,
            }
            with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            
            # 保存图像
            if "baseline" in results:
                results["baseline"].save(sample_dir / "baseline.png")
            if "adapter" in results:
                results["adapter"].save(sample_dir / "adapter.png")
            if "annotated" in results:
                results["annotated"].save(sample_dir / "annotated.png")
            
            # 创建对比图
            if "baseline" in results and "adapter" in results:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(caption[:80], fontsize=12, fontweight="bold")
                
                axes[0].imshow(results["baseline"])
                axes[0].set_title("Baseline (No Adapter)")
                axes[0].axis("off")
                
                axes[1].imshow(results["adapter"])
                axes[1].set_title("With Adapter")
                axes[1].axis("off")
                
                axes[2].imshow(results["annotated"])
                axes[2].set_title("Annotated (Target Positions)")
                axes[2].axis("off")
                
                plt.tight_layout()
                plt.savefig(sample_dir / "comparison.png", dpi=150, bbox_inches="tight")
                plt.close()
            
            print(f"  ✓ 保存到: {sample_dir}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 60)
    print("验证完成！")
    print("=" * 60)
    print(f"成功样本: {success_count}/{len(test_data)}")
    print(f"输出目录: {output_dir}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证第一阶段训练效果（修复版）")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--kolors-path", type=str, default="./model/Kolors", help="Kolors 模型路径")
    parser.add_argument("--test-data", type=str, default="./data/coco2014_cn_val_clean.jsonl", help="测试数据路径")
    parser.add_argument("--num-samples", type=int, default=5, help="测试样本数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--resolution", type=int, default=512, help="生成分辨率")
    
    args = parser.parse_args()
    
    validate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        kolors_path=args.kolors_path,
        test_data_path=args.test_data,
        num_samples=args.num_samples,
        device=args.device,
        resolution=args.resolution,
    )

