#!/usr/bin/env python3
"""
准备反馈验证数据集

生成一批测试图像（使用当前模型），然后使用 Qwen-VL 或 GPT-4V 验证位置准确性，
构建 (生成图像, 原始prompt, 验证结果, 修正prompt) 四元组。

Usage:
    python scripts/prepare_feedback_dataset.py \
        --model-path checkpoints/gill_opt/ \
        --prompts-file data/test_prompts.txt \
        --output-jsonl data/feedback_dataset.jsonl \
        --num-samples 1000
"""

import argparse
import json
import os
from typing import Dict, List
from tqdm import tqdm
from PIL import Image
import torch

from gill import models
from gill import feedback_verifier


def generate_test_images(gill_model, prompts: List[str], output_dir: str) -> List[str]:
    """
    使用 GILL 模型生成测试图像
    
    Args:
        gill_model: GILL 模型实例
        prompts: prompt 列表
        output_dir: 输出目录
    
    Returns:
        image_paths: 生成的图像路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    for i, prompt in enumerate(tqdm(prompts, desc="生成图像")):
        try:
            # 生成图像（带布局控制）
            outputs = gill_model.generate_for_images_and_texts(
                prompts=[prompt],
                num_words=32,
                min_word_tokens=5,
                ret_scale_factor=1.0,
                gen_scale_factor=1.0,
                max_num_rets=1
            )
            
            # 提取生成的图像
            for output in outputs:
                if isinstance(output, dict) and 'gen' in output:
                    gen_images = output['gen']
                    if gen_images:
                        # 保存第一张生成的图像
                        image = gen_images[0][0] if isinstance(gen_images[0], tuple) else gen_images[0]
                        image_path = os.path.join(output_dir, f"gen_{i:05d}.png")
                        image.save(image_path)
                        image_paths.append(image_path)
                        break
        except Exception as e:
            print(f"⚠️ 生成图像 {i} 失败: {e}")
            continue
    
    return image_paths


def verify_images(verifier, image_paths: List[str], prompts: List[str],
                  expected_layouts: List[List[Dict]] = None) -> List[Dict]:
    """
    验证生成的图像
    
    Args:
        verifier: FeedbackVerifier 实例
        image_paths: 图像路径列表
        prompts: 对应的 prompt 列表
        expected_layouts: 期望的布局列表（可选）
    
    Returns:
        verification_results: 验证结果列表
    """
    results = []
    
    for image_path, prompt in tqdm(zip(image_paths, prompts), 
                                   total=len(image_paths), 
                                   desc="验证图像"):
        try:
            image = Image.open(image_path).convert('RGB')
            expected_layout = expected_layouts[image_paths.index(image_path)] if expected_layouts else None
            
            result = verifier.verify(image, prompt, expected_layout)
            result['image_path'] = image_path
            result['original_prompt'] = prompt
            results.append(result)
        except Exception as e:
            print(f"⚠️ 验证图像 {image_path} 失败: {e}")
            results.append({
                "correct": False,
                "confidence": 0.0,
                "feedback": f"验证失败: {str(e)}",
                "suggested_prompt": prompt,
                "image_path": image_path,
                "original_prompt": prompt
            })
    
    return results


def create_feedback_dataset(model_path: str,
                           prompts_file: str,
                           output_jsonl: str,
                           num_samples: int = 1000,
                           output_image_dir: str = "feedback_images"):
    """
    创建反馈数据集
    
    Args:
        model_path: GILL 模型路径
        prompts_file: prompt 文件路径（每行一个 prompt）
        output_jsonl: 输出 JSONL 文件路径
        num_samples: 样本数量
        output_image_dir: 生成的图像保存目录
    """
    # 1. 加载模型
    print(f"📦 加载 GILL 模型: {model_path}")
    gill_model = models.load_gill(model_path, load_ret_embs=False, load_sd=True)
    gill_model.eval()
    
    # 2. 加载 prompts
    print(f"📖 读取 prompts: {prompts_file}")
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()][:num_samples]
    
    print(f"✓ 共 {len(prompts)} 个 prompts")
    
    # 3. 生成测试图像
    print(f"\n🎨 生成测试图像...")
    image_paths = generate_test_images(gill_model, prompts, output_image_dir)
    print(f"✓ 生成 {len(image_paths)} 张图像")
    
    # 4. 验证图像
    print(f"\n🔍 验证图像...")
    verifier = feedback_verifier.create_feedback_verifier()
    verification_results = verify_images(verifier, image_paths, prompts)
    
    # 5. 保存结果
    print(f"\n💾 保存反馈数据集: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for result in verification_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 统计信息
    correct_count = sum(1 for r in verification_results if r.get("correct", False))
    print(f"\n📊 统计信息:")
    print(f"  总样本数: {len(verification_results)}")
    print(f"  通过验证: {correct_count} ({correct_count/len(verification_results)*100:.1f}%)")
    print(f"  未通过验证: {len(verification_results) - correct_count}")
    
    print(f"✓ 完成！")


def main():
    parser = argparse.ArgumentParser(description="准备反馈验证数据集")
    parser.add_argument('--model-path', type=str, required=True,
                       help='GILL 模型路径')
    parser.add_argument('--prompts-file', type=str, required=True,
                       help='Prompt 文件路径（每行一个）')
    parser.add_argument('--output-jsonl', type=str, required=True,
                       help='输出 JSONL 文件路径')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='样本数量')
    parser.add_argument('--output-image-dir', type=str, default='feedback_images',
                       help='生成的图像保存目录')
    
    args = parser.parse_args()
    
    create_feedback_dataset(
        args.model_path,
        args.prompts_file,
        args.output_jsonl,
        args.num_samples,
        args.output_image_dir
    )


if __name__ == '__main__':
    main()

