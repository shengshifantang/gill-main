#!/usr/bin/env python3
"""
从数据集中创建测试集（用于 Baseline 对比）

确保测试集包含：
1. 简单场景（1-2 个物体）
2. 复杂场景（3-5 个物体）
3. 反直觉场景（如"把大象放进冰箱"）
4. 复杂空间关系（如"左上角是A，右下角是B，中间是C"）

Usage:
    python scripts/create_test_set.py \
        --input-jsonl data/layout_dataset_final_15k.jsonl \
        --output-jsonl data/test_set_baseline.jsonl \
        --num-samples 500 \
        --stratify
"""

import argparse
import json
import os
import random
from typing import List, Dict
from collections import defaultdict


def classify_complexity(sample: Dict) -> str:
    """根据样本复杂度分类"""
    objects = sample.get("objects", [])
    num_objects = len(objects)
    caption = sample.get("caption", "").lower()
    
    # 反直觉场景关键词
    counter_intuitive_keywords = [
        "大象", "冰箱", "放进", "装进", "塞进",
        "巨大", "微小", "不可能", "荒谬"
    ]
    
    if any(kw in caption for kw in counter_intuitive_keywords):
        return "counter_intuitive"
    
    if num_objects <= 2:
        return "simple"
    elif num_objects <= 5:
        return "medium"
    else:
        return "complex"


def create_stratified_test_set(input_file: str, output_file: str, num_samples: int = 500):
    """创建分层测试集"""
    
    # 加载数据
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"📊 加载数据: {len(samples)} 条")
    
    # 按复杂度分类
    classified = defaultdict(list)
    for sample in samples:
        complexity = classify_complexity(sample)
        classified[complexity].append(sample)
    
    print(f"📊 分类统计:")
    for k, v in classified.items():
        print(f"  - {k}: {len(v)} 条")
    
    # 分层采样
    test_samples = []
    
    # 简单场景: 30%
    simple_count = int(num_samples * 0.3)
    if len(classified["simple"]) >= simple_count:
        test_samples.extend(random.sample(classified["simple"], simple_count))
    
    # 中等场景: 40%
    medium_count = int(num_samples * 0.4)
    if len(classified["medium"]) >= medium_count:
        test_samples.extend(random.sample(classified["medium"], medium_count))
    
    # 复杂场景: 20%
    complex_count = int(num_samples * 0.2)
    if len(classified["complex"]) >= complex_count:
        test_samples.extend(random.sample(classified["complex"], complex_count))
    
    # 反直觉场景: 10%
    counter_count = int(num_samples * 0.1)
    if len(classified["counter_intuitive"]) >= counter_count:
        test_samples.extend(random.sample(classified["counter_intuitive"], counter_count))
    else:
        # 如果反直觉场景不够，从复杂场景补充
        remaining = num_samples - len(test_samples)
        if remaining > 0 and len(classified["complex"]) > complex_count:
            test_samples.extend(random.sample(
                [s for s in classified["complex"] if s not in test_samples],
                min(remaining, len(classified["complex"]) - complex_count)
            ))
    
    # 打乱顺序
    random.shuffle(test_samples)
    
    # 添加 ID
    for i, sample in enumerate(test_samples):
        sample["id"] = i
        sample["complexity"] = classify_complexity(sample)
    
    # 保存
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 测试集已创建: {len(test_samples)} 条")
    print(f"   输出文件: {output_file}")
    
    # 统计
    complexity_dist = defaultdict(int)
    for sample in test_samples:
        complexity_dist[sample["complexity"]] += 1
    
    print(f"\n📊 测试集分布:")
    for k, v in complexity_dist.items():
        print(f"  - {k}: {v} 条 ({v/len(test_samples)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="创建测试集")
    parser.add_argument("--input-jsonl", type=str, required=True,
                       help="输入数据集 JSONL")
    parser.add_argument("--output-jsonl", type=str, required=True,
                       help="输出测试集 JSONL")
    parser.add_argument("--num-samples", type=int, default=500,
                       help="测试集样本数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    create_stratified_test_set(args.input_jsonl, args.output_jsonl, args.num_samples)


if __name__ == "__main__":
    main()

