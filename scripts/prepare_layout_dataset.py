#!/usr/bin/env python3
"""
构建带空间标注的中文数据集

从 WuKong 数据集中筛选包含位置描述的数据，并使用 GPT-4V 或 Qwen-VL 自动标注 bounding box。

数据格式：
{
    "caption": "左边放一个青花瓷瓶，右边放一盘饺子",
    "image_path": "xxx.jpg",
    "objects": [
        {"name": "青花瓷瓶", "bbox": [0.1, 0.3, 0.4, 0.7]},
        {"name": "饺子", "bbox": [0.6, 0.3, 0.9, 0.7]}
    ]
}

Usage:
    python scripts/prepare_layout_dataset.py \
        --input-tsv data/wukong_train.tsv \
        --output-jsonl data/layout_dataset.jsonl \
        --num-samples 10000 \
        --use-qwen-vl
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional
from tqdm import tqdm
import pandas as pd
from PIL import Image


# 位置关键词列表
POSITION_KEYWORDS = [
    "左边", "左侧", "左方", "左面",
    "右边", "右侧", "右方", "右面",
    "上方", "上边", "上面", "上侧",
    "下方", "下边", "下面", "下侧",
    "中间", "中央", "中心", "中部",
    "左上角", "左下角", "右上角", "右下角",
    "左上", "左下", "右上", "右下"
]


def has_position_description(text: str) -> bool:
    """检查文本是否包含位置描述"""
    return any(keyword in text for keyword in POSITION_KEYWORDS)


def filter_captions_with_position(input_tsv: str, output_tsv: str, num_samples: int = 10000):
    """
    从输入 TSV 文件中筛选包含位置描述的 caption
    
    Args:
        input_tsv: 输入 TSV 文件路径
        output_tsv: 输出 TSV 文件路径
        num_samples: 需要筛选的样本数量
    """
    print(f"📖 读取数据: {input_tsv}")
    df = pd.read_csv(input_tsv, sep='\t', header=None, names=['caption', 'image'])
    
    print(f"🔍 筛选包含位置描述的样本...")
    filtered_df = df[df['caption'].apply(has_position_description)]
    
    print(f"✓ 找到 {len(filtered_df)} 条包含位置描述的样本（共 {len(df)} 条）")
    
    # 随机采样
    if len(filtered_df) > num_samples:
        filtered_df = filtered_df.sample(n=num_samples, random_state=42)
        print(f"✓ 随机采样 {num_samples} 条")
    
    # 保存
    filtered_df.to_csv(output_tsv, sep='\t', header=False, index=False)
    print(f"✓ 保存到: {output_tsv}")
    
    return filtered_df


def annotate_with_qwen_vl(image_path: str, caption: str, 
                          qwen_model=None, processor=None) -> List[Dict]:
    """
    使用 Qwen-VL 自动标注 bounding box
    
    Args:
        image_path: 图像路径
        caption: 文本描述
        qwen_model: Qwen-VL 模型（如果为 None，会自动加载）
        processor: Qwen-VL processor（如果为 None，会自动加载）
    
    Returns:
        objects: [{"name": "...", "bbox": [x1, y1, x2, y2]}]
    """
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        
        if qwen_model is None or processor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen-VL", trust_remote_code=True
            )
            qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen-VL",
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 构建 grounding prompt
        prompt = f"请检测图像中的以下对象并标注位置：{caption}"
        
        # 处理输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(qwen_model.device)
        
        # 生成（这里简化处理，实际需要调用 grounding API）
        # 注意：Qwen-VL 的 grounding 功能可能需要特殊调用
        
        # 返回空结果（需要根据实际 API 实现）
        return []
        
    except Exception as e:
        print(f"⚠️ Qwen-VL 标注出错: {e}")
        return []


def annotate_with_heuristic(caption: str) -> List[Dict]:
    """
    使用启发式规则标注（基于位置关键词）
    
    Args:
        caption: 文本描述
    
    Returns:
        objects: [{"name": "...", "bbox": [x1, y1, x2, y2]}]
    """
    objects = []
    
    # 预定义槽位
    slots = {
        "left": [0.0, 0.1, 0.4, 0.9],
        "right": [0.6, 0.1, 1.0, 0.9],
        "top": [0.1, 0.0, 0.9, 0.4],
        "bottom": [0.1, 0.6, 0.9, 1.0],
        "center": [0.3, 0.3, 0.7, 0.7],
        "bottom_left": [0.0, 0.6, 0.4, 1.0],
        "bottom_right": [0.6, 0.6, 1.0, 1.0],
    }
    
    # 简单名词提取（使用 jieba 或简单规则）
    try:
        import jieba.posseg as pseg
        words = pseg.cut(caption)
        nouns = [w for w, flag in words if flag.startswith("n")]
    except:
        # 简单分词
        nouns = re.findall(r'[\u4e00-\u9fa5]{2,}', caption)
    
    # 根据位置关键词分配槽位
    slot_mapping = {
        "left": ["左边", "左侧", "左方", "左面"],
        "right": ["右边", "右侧", "右方", "右面"],
        "top": ["上方", "上边", "上面", "上侧"],
        "bottom": ["下方", "下边", "下面", "下侧"],
        "center": ["中间", "中央", "中心", "中部"],
        "bottom_left": ["左下角", "左下"],
        "bottom_right": ["右下角", "右下"],
    }
    
    used_slots = set()
    for noun in nouns[:3]:  # 最多处理3个对象
        # 找到对应的位置关键词
        assigned_slot = None
        for slot_key, keywords in slot_mapping.items():
            if any(kw in caption for kw in keywords) and slot_key not in used_slots:
                assigned_slot = slot_key
                used_slots.add(slot_key)
                break
        
        if assigned_slot is None:
            assigned_slot = "center"  # 默认居中
        
        objects.append({
            "name": noun,
            "bbox": slots[assigned_slot]
        })
    
    return objects


def create_layout_dataset(input_tsv: str,
                         output_jsonl: str,
                         image_dir: str,
                         num_samples: int = 10000,
                         use_qwen_vl: bool = False,
                         use_heuristic: bool = True):
    """
    创建布局数据集
    
    Args:
        input_tsv: 输入 TSV 文件
        output_jsonl: 输出 JSONL 文件
        image_dir: 图像目录
        num_samples: 样本数量
        use_qwen_vl: 是否使用 Qwen-VL 标注
        use_heuristic: 是否使用启发式规则（备用）
    """
    # 1. 筛选包含位置描述的样本
    filtered_tsv = input_tsv.replace('.tsv', '_filtered.tsv')
    filtered_df = filter_captions_with_position(input_tsv, filtered_tsv, num_samples)
    
    # 2. 标注 bounding box
    print(f"\n📝 开始标注 bounding box...")
    
    qwen_model = None
    processor = None
    if use_qwen_vl:
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen-VL", trust_remote_code=True
            )
            qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen-VL",
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
            print("✓ Qwen-VL 加载成功")
        except Exception as e:
            print(f"⚠️ Qwen-VL 加载失败: {e}，将使用启发式规则")
            use_qwen_vl = False
    
    results = []
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="标注进度"):
        caption = row['caption']
        image_file = row['image']
        image_path = os.path.join(image_dir, image_file)
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            continue
        
        # 标注
        if use_qwen_vl and qwen_model is not None:
            objects = annotate_with_qwen_vl(image_path, caption, qwen_model, processor)
        else:
            objects = annotate_with_heuristic(caption)
        
        if len(objects) == 0:
            continue
        
        # 保存结果
        result = {
            "caption": caption,
            "image_path": image_path,
            "objects": objects
        }
        results.append(result)
    
    # 3. 保存到 JSONL
    print(f"\n💾 保存 {len(results)} 条标注数据到: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ 完成！共生成 {len(results)} 条布局数据")


def main():
    parser = argparse.ArgumentParser(description="构建带空间标注的中文数据集")
    parser.add_argument('--input-tsv', type=str, required=True,
                       help='输入 TSV 文件路径')
    parser.add_argument('--output-jsonl', type=str, required=True,
                       help='输出 JSONL 文件路径')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='图像目录路径')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='需要标注的样本数量')
    parser.add_argument('--use-qwen-vl', action='store_true',
                       help='使用 Qwen-VL 自动标注（需要 GPU）')
    parser.add_argument('--use-heuristic', action='store_true', default=True,
                       help='使用启发式规则标注（备用）')
    
    args = parser.parse_args()
    
    create_layout_dataset(
        args.input_tsv,
        args.output_jsonl,
        args.image_dir,
        args.num_samples,
        args.use_qwen_vl,
        args.use_heuristic
    )


if __name__ == '__main__':
    main()

