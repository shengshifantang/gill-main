#!/usr/bin/env python3
"""
Spatial Adapter 推理 Demo

加载训练好的 Adapter，验证布局控制效果。

Usage:
    python scripts/inference_demo.py \
        --kolors-model ./model/Kolors \
        --adapter-path ./checkpoints/spatial_adapter_wukong_v2/spatial_adapter_final.pt \
        --prompt "一只猫在左边，一只狗在右边，草地背景" \
        --output demo_result.png
"""

import argparse
import os
import sys
import torch
from PIL import Image, ImageDraw
from diffusers import KolorsPipeline

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.spatial_adapter import (
    inject_spatial_control_to_unet,
    remove_spatial_control_from_unet,
    create_spatial_adapter_for_kolors,
)


def fix_kolors_tokenizer(pipeline):
    """修复 Kolors Tokenizer 的 padding_side 兼容性问题"""
    original_pad = pipeline.tokenizer._pad
    
    def patched_pad(*args, **kwargs):
        # 移除 padding_side 参数（如果存在）
        kwargs.pop('padding_side', None)
        return original_pad(*args, **kwargs)
    
    pipeline.tokenizer._pad = patched_pad
    print("✓ 已修复 Kolors Tokenizer padding 兼容性")


def draw_boxes(image: Image.Image, boxes: list, color: str = "red", width: int = 5):
    """
    在图像上绘制边界框
    
    Args:
        image: PIL Image
        boxes: List of [x1, y1, x2, y2] 归一化坐标 (0-1)
        color: 框的颜色
        width: 框的宽度
    """
    draw = ImageDraw.Draw(image)
    W, H = image.size
    
    for box in boxes:
        # 归一化坐标转换为像素坐标
        x1, y1, x2, y2 = box
        x1_px = x1 * W
        y1_px = y1 * H
        x2_px = x2 * W
        y2_px = y2 * H
        
        # 绘制矩形框
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline=color, width=width)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Spatial Adapter 推理 Demo")
    parser.add_argument(
        "--kolors-model",
        type=str,
        default="./model/Kolors",
        help="Kolors 模型路径"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="训练好的 Adapter 权重路径"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="一只猫在左边，一只狗在右边，草地背景",
        help="生成提示词"
    )
    parser.add_argument(
        "--boxes",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.8, 0.6, 0.2, 1.0, 0.8],
        help="边界框坐标 [x1, y1, x2, y2, ...] (归一化 0-1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_result.png",
        help="输出图像路径"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="生成图像高度"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="生成图像宽度"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Guidance scale"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Classifier-Free Guidance 的负提示词，留空则使用空串"
    )
    parser.add_argument(
        "--gate-scale",
        type=float,
        default=1.0,
        help="推理时放大 gate（>1 增强空间约束，谨慎使用，默认 1.0）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，便于复现；设为 -1 则使用随机种子"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # 1. 加载 Kolors Pipeline
    print(f"🚀 加载 Kolors Pipeline: {args.kolors_model}")
    device = args.device if torch.cuda.is_available() else "cpu"
    
    pipeline = KolorsPipeline.from_pretrained(
        args.kolors_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # 部分 diffusers 版本在本地权重缺少 fp16 目录时会因 variant 报错，直接省略
        trust_remote_code=True
    ).to(device)
    
    # 修复 Tokenizer 兼容性
    fix_kolors_tokenizer(pipeline)
    
    # 2. 加载 Adapter
    print(f"📦 加载 Adapter: {args.adapter_path}")
    if not os.path.exists(args.adapter_path):
        raise FileNotFoundError(f"Adapter 文件不存在: {args.adapter_path}")
    
    # 创建 Adapter 容器（动态维度管理）
    adapter_container = create_spatial_adapter_for_kolors()
    
    # 加载权重
    state_dict = torch.load(args.adapter_path, map_location=device)
    
    # 动态补齐容器里需要的维度，再加载（strict=False 忽略冗余键）
    if isinstance(state_dict, dict) and any(k.startswith("dim_") for k in state_dict.keys()):
        # 解析 state_dict 中包含的维度
        dims_in_ckpt = set()
        for k in state_dict.keys():
            if k.startswith("dim_"):
                try:
                    dim_val = int(k.split("_")[1].split(".")[0])
                    dims_in_ckpt.add(dim_val)
                except Exception:
                    continue
        # 为每个维度确保容器里有对应 Adapter
        for d in sorted(dims_in_ckpt):
            key = f"dim_{d}"
            if key not in adapter_container:
                # Kolors 的 UNet 典型维度有 320/640/1280/2048 等，统一创建
                adapter_container[key] = adapter_container["dim_2048"].__class__(hidden_dim=d, num_heads=8)
        adapter_container.load_state_dict(state_dict, strict=False)
    else:
        # 如果是单个 Adapter 的权重，需要适配到容器格式
        # 假设是默认维度 2048
        if "dim_2048" in adapter_container.state_dict():
            adapter_container["dim_2048"].load_state_dict(state_dict)
        else:
            # 尝试直接加载到容器
            adapter_container.load_state_dict(state_dict, strict=False)
    
    # Adapter 保持 FP32（训练时就是 FP32）
    adapter_container = adapter_container.to(device=device, dtype=torch.float32)
    print("✓ Adapter 加载完成")
    
    # 可选：放大 gate，加强空间约束（临时 hack，过大会影响画质）
    if args.gate_scale != 1.0:
        with torch.no_grad():
            scaled_cnt = 0
            for m in adapter_container.modules():
                if hasattr(m, "gate"):
                    m.gate.mul_(args.gate_scale)
                    scaled_cnt += 1
        print(f"✓ gate 放大系数 {args.gate_scale} 已应用，作用层数: {scaled_cnt}")
    
    # 设置随机种子，保证复现；seed=-1 时不固定
    generator = None
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"✓ 已设置随机种子: {args.seed}")
    
    # 3. 准备 BBoxes
    boxes_list = args.boxes
    if len(boxes_list) % 4 != 0:
        raise ValueError(f"BBoxes 坐标数量必须是 4 的倍数，当前: {len(boxes_list)}")
    
    # 转换为 (B, N, 4) 格式
    num_boxes = len(boxes_list) // 4
    boxes = [[boxes_list[i*4], boxes_list[i*4+1], boxes_list[i*4+2], boxes_list[i*4+3]] 
             for i in range(num_boxes)]
    
    bboxes_tensor = torch.tensor([boxes], device=device, dtype=torch.float32)
    print(f"✓ 准备 {num_boxes} 个边界框: {boxes}")
    
    # 4. 注入 Spatial Control
    print("🔧 注入 Spatial Control 到 UNet...")
    orig_procs, spatial_procs, adapter_container = inject_spatial_control_to_unet(
        pipeline.unet,
        adapter_dict=adapter_container,
        bboxes=bboxes_tensor
    )
    print("✓ Spatial Control 已注入")
    
    try:
        # 5. 生成图像
        print(f"🎨 生成图像: {args.prompt}")
        print(f"   尺寸: {args.width}x{args.height}")
        print(f"   步数: {args.num_inference_steps}, Guidance: {args.guidance_scale}")
        
        image = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        
        print("✓ 图像生成完成")
        
        # 6. 绘制边界框
        image_with_boxes = draw_boxes(image.copy(), boxes)
        
        # 7. 保存结果
        image_with_boxes.save(args.output)
        print(f"✅ 结果已保存至: {args.output}")
        
        # 同时保存不带框的原始图像
        if args.output.endswith('.png'):
            raw_output = args.output.replace('.png', '_raw.png')
        else:
            raw_output = args.output + '_raw.png'
        image.save(raw_output)
        print(f"✅ 原始图像已保存至: {raw_output}")
        
    finally:
        # 8. 移除 Spatial Control（恢复原始 UNet）
        print("🔧 移除 Spatial Control...")
        remove_spatial_control_from_unet(pipeline.unet, orig_procs)
        print("✓ Spatial Control 已移除")


if __name__ == "__main__":
    main()

