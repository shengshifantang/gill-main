#!/usr/bin/env python3
"""
准备混合训练数据（15k 黄金数据 + 50k 通用数据）

策略：
- 15k Layout Data: 带精确 BBox 的黄金数据
- 50k General Data: 从 Wukong 随机抽取，无 BBox（或全图 BBox）

用于防止过拟合和灾难性遗忘。

Usage:
    python scripts/prepare_mixed_training_data.py \
        --layout-data data/layout_dataset_final_15k.jsonl \
        --general-data data/wukong_release \
        --output-jsonl data/mixed_training_65k.jsonl \
        --layout-ratio 0.3 \
        --general-count 50000
"""

import argparse
import json
import os
import random
import pandas as pd
from typing import List, Dict
from tqdm import tqdm


def load_layout_data(layout_file: str) -> List[Dict]:
    """加载带布局的黄金数据"""
    samples = []
    with open(layout_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_general_data(wukong_dir: str, num_samples: int) -> List[Dict]:
    """从 Wukong 数据集中随机抽取通用数据（无布局）"""
    samples = []
    
    # 收集所有 CSV 文件
    csv_files = []
    if os.path.isdir(wukong_dir):
        for root, dirs, files in os.walk(wukong_dir):
            for f in files:
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))
    elif os.path.isfile(wukong_dir):
        csv_files = [wukong_dir]
    
    if not csv_files:
        print(f"⚠️ 未找到 Wukong 数据文件: {wukong_dir}")
        return samples
    
    print(f"📊 从 {len(csv_files)} 个 CSV 文件中抽取通用数据...")
    
    # 随机打乱文件顺序
    random.shuffle(csv_files)
    
    # 从多个文件中抽取
    chunk_size = 10000
    for csv_file in csv_files:
        if len(samples) >= num_samples:
            break
        
        try:
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, encoding='utf-8'):
                if len(samples) >= num_samples:
                    break
                
                for _, row in chunk.iterrows():
                    if len(samples) >= num_samples:
                        break
                    
                    caption = str(row.get('caption', '') or row.get('text', '')).strip()
                    if not caption or len(caption) < 5:
                        continue
                    
                    # 构造通用数据样本（无布局信息）
                    sample = {
                        'caption': caption,
                        'url': str(row.get('url', '')),
                        'image_path': str(row.get('image_path', row.get('image', ''))),
                        'has_layout': False,  # 标记为无布局数据
                        'objects': []  # 空对象列表
                    }
                    
                    samples.append(sample)
        except Exception as e:
            print(f"⚠️ 处理文件 {csv_file} 时出错: {e}")
            continue
    
    print(f"✓ 抽取了 {len(samples)} 条通用数据")
    return samples


def create_mixed_dataset(layout_samples: List[Dict], general_samples: List[Dict],
                         layout_ratio: float = 0.3) -> List[Dict]:
    """
    创建混合数据集
    
    Args:
        layout_samples: 带布局的黄金数据
        general_samples: 通用数据（无布局）
        layout_ratio: Layout 数据在 batch 中的比例（例如 0.3 表示 30% Layout, 70% General）
    
    Returns:
        混合后的数据集
    """
    # 标记数据来源
    for sample in layout_samples:
        sample['has_layout'] = True
        sample['data_source'] = 'layout_golden'
    
    for sample in general_samples:
        sample['has_layout'] = False
        sample['data_source'] = 'general_wukong'
    
    # 计算混合比例
    total_layout = len(layout_samples)
    # 根据 layout_ratio 计算需要的 general 数据量
    # layout_ratio = layout / (layout + general)
    # general = layout * (1 - layout_ratio) / layout_ratio
    if layout_ratio > 0:
        target_general = int(total_layout * (1 - layout_ratio) / layout_ratio)
    else:
        target_general = len(general_samples)
    
    # 如果 general 数据不够，使用全部
    if len(general_samples) < target_general:
        target_general = len(general_samples)
        print(f"⚠️ 通用数据不足，使用全部 {len(general_samples)} 条")
    
    # 随机采样 general 数据
    selected_general = random.sample(general_samples, min(target_general, len(general_samples)))
    
    # 合并
    mixed_samples = layout_samples + selected_general
    
    # 打乱顺序
    random.shuffle(mixed_samples)
    
    # 添加混合信息
    for i, sample in enumerate(mixed_samples):
        sample['mixed_id'] = i
        sample['is_layout'] = sample.get('has_layout', False)
    
    return mixed_samples


def main():
    parser = argparse.ArgumentParser(description="准备混合训练数据")
    parser.add_argument(
        "--layout-data",
        type=str,
        required=True,
        help="带布局的黄金数据 JSONL 文件",
    )
    parser.add_argument(
        "--general-data",
        type=str,
        required=True,
        help="Wukong 数据目录或文件（用于抽取通用数据）",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="输出的混合数据集 JSONL 文件",
    )
    parser.add_argument(
        "--layout-ratio",
        type=float,
        default=0.3,
        help="Layout 数据在混合数据集中的比例（0.3 表示 30% Layout, 70% General）",
    )
    parser.add_argument(
        "--general-count",
        type=int,
        default=50000,
        help="从 Wukong 中抽取的通用数据数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("🔄 准备混合训练数据")
    print("=" * 60)
    print(f"Layout 数据: {args.layout_data}")
    print(f"通用数据源: {args.general_data}")
    print(f"Layout 比例: {args.layout_ratio}")
    print(f"通用数据量: {args.general_count}")
    print()
    
    # 1. 加载 Layout 数据
    print("📥 Step 1: 加载 Layout 数据...")
    layout_samples = load_layout_data(args.layout_data)
    print(f"✓ 加载 {len(layout_samples)} 条 Layout 数据")
    
    # 2. 加载通用数据
    print(f"\n📥 Step 2: 从 Wukong 抽取通用数据...")
    general_samples = load_general_data(args.general_data, args.general_count)
    print(f"✓ 抽取 {len(general_samples)} 条通用数据")
    
    # 3. 创建混合数据集
    print(f"\n🔄 Step 3: 创建混合数据集...")
    mixed_samples = create_mixed_dataset(
        layout_samples,
        general_samples,
        args.layout_ratio
    )
    
    # 统计
    layout_count = sum(1 for s in mixed_samples if s.get('has_layout', False))
    general_count = len(mixed_samples) - layout_count
    
    print(f"✓ 混合完成:")
    print(f"  - Layout 数据: {layout_count} 条 ({layout_count/len(mixed_samples)*100:.1f}%)")
    print(f"  - 通用数据: {general_count} 条 ({general_count/len(mixed_samples)*100:.1f}%)")
    print(f"  - 总计: {len(mixed_samples)} 条")
    
    # 4. 保存
    print(f"\n💾 Step 4: 保存混合数据集...")
    os.makedirs(os.path.dirname(args.output_jsonl) if os.path.dirname(args.output_jsonl) else '.', exist_ok=True)
    
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for sample in mixed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 混合数据集已保存: {args.output_jsonl}")
    print()
    print("💡 提示: 在训练时，可以根据 'has_layout' 字段决定是否使用布局控制")


if __name__ == "__main__":
    main()

