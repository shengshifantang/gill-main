#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全的 Layout Planner 推理包装器

提供实体检测和空输出保护，避免模型在无对象场景下强行输出布局。
"""

import re
from typing import Dict, List, Optional


def detect_entities(text: str, min_length: int = 1) -> List[str]:
    """
    检测文本中的实体（名词）
    
    Args:
        text: 输入文本
        min_length: 最小实体长度
    
    Returns:
        实体列表
    """
    try:
        import jieba.posseg as pseg
        
        # 使用 jieba 词性标注
        words = pseg.cut(text)
        nouns = []
        
        for word, flag in words:
            # 名词类：n, nr (人名), ns (地名), nt (机构名), nz (其他专名)
            if flag.startswith('n') and len(word) >= min_length:
                nouns.append(word)
        
        return nouns
    
    except ImportError:
        # 如果 jieba 未安装，使用简单的启发式方法
        # 提取中文词汇（长度 >= min_length）
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', text)
        return [w for w in chinese_words if len(w) >= min_length]


def has_spatial_keywords(text: str) -> bool:
    """
    检测文本中是否包含空间位置关键词
    
    Args:
        text: 输入文本
    
    Returns:
        是否包含空间关键词
    """
    spatial_keywords = [
        # 方位词
        "左边", "左侧", "左方", "右边", "右侧", "右方",
        "上方", "上边", "上面", "下方", "下边", "下面",
        "中间", "中央", "中心", "旁边", "附近",
        
        # 角落
        "左上角", "右上角", "左下角", "右下角",
        
        # 前后
        "前面", "后面", "前方", "后方", "背景", "前景",
        
        # 相对位置
        "之间", "之上", "之下", "之左", "之右"
    ]
    
    return any(keyword in text for keyword in spatial_keywords)


def safe_generate_layout(
    planner,
    caption: str,
    min_entities: int = 1,
    require_spatial_keywords: bool = False,
    **generate_kwargs
) -> Dict:
    """
    安全的布局生成，带实体检测和空输出保护
    
    Args:
        planner: LayoutPlanner 实例
        caption: 输入文本
        min_entities: 最少实体数（低于此值则不输出布局）
        require_spatial_keywords: 是否要求包含空间关键词
        **generate_kwargs: 传递给 planner.generate_layout 的其他参数
    
    Returns:
        布局生成结果，如果不满足条件则返回空布局
    """
    # 1. 检测实体
    entities = detect_entities(caption)
    
    # 2. 检测空间关键词
    has_spatial = has_spatial_keywords(caption)
    
    # 3. 判断是否应该生成布局
    should_generate = True
    skip_reason = None
    
    if len(entities) < min_entities:
        should_generate = False
        skip_reason = f"实体数 ({len(entities)}) < {min_entities}"
    
    if require_spatial_keywords and not has_spatial:
        should_generate = False
        skip_reason = "缺少空间位置关键词"
    
    # 4. 如果不应该生成，返回空布局
    if not should_generate:
        print(f"⚠️ 跳过布局生成: {skip_reason}")
        print(f"   Caption: {caption}")
        print(f"   检测到的实体: {entities}")
        return {
            "layout_text": "",
            "objects": [],
            "skip_reason": skip_reason,
            "detected_entities": entities
        }
    
    # 5. 生成布局
    result = planner.generate_layout(caption, **generate_kwargs)
    
    # 6. 后验证：如果模型输出了对象，但实体数不匹配，给出警告
    if result['objects']:
        num_generated = len(result['objects'])
        num_entities = len(entities)
        
        if num_generated > num_entities * 2:
            print(f"⚠️ 警告: 生成的对象数 ({num_generated}) 远超实体数 ({num_entities})")
            print(f"   可能存在过度生成，建议检查")
    
    result['detected_entities'] = entities
    return result


def batch_safe_generate_layout(
    planner,
    captions: List[str],
    min_entities: int = 1,
    require_spatial_keywords: bool = False,
    **generate_kwargs
) -> List[Dict]:
    """
    批量安全生成布局
    
    Args:
        planner: LayoutPlanner 实例
        captions: 输入文本列表
        min_entities: 最少实体数
        require_spatial_keywords: 是否要求包含空间关键词
        **generate_kwargs: 传递给 planner.generate_layout 的其他参数
    
    Returns:
        布局生成结果列表
    """
    results = []
    
    for caption in captions:
        result = safe_generate_layout(
            planner=planner,
            caption=caption,
            min_entities=min_entities,
            require_spatial_keywords=require_spatial_keywords,
            **generate_kwargs
        )
        results.append(result)
    
    return results


# 使用示例
if __name__ == "__main__":
    # 测试实体检测
    test_cases = [
        "桌子左边有一只猫",           # 有实体 + 空间关键词 ✓
        "美丽的风景",                 # 无明确实体 ✗
        "蓝天白云",                   # 有实体但无空间关键词 ?
        "一个人在跑步",               # 有实体但无空间关键词 ?
        "抽象的艺术作品",             # 无明确实体 ✗
        "左边是树，右边是房子",       # 有实体 + 空间关键词 ✓
    ]
    
    print("=" * 60)
    print("实体检测测试")
    print("=" * 60)
    
    for caption in test_cases:
        entities = detect_entities(caption)
        has_spatial = has_spatial_keywords(caption)
        
        print(f"\nCaption: {caption}")
        print(f"  实体: {entities} ({len(entities)} 个)")
        print(f"  空间关键词: {'✓' if has_spatial else '✗'}")
        print(f"  建议生成布局: {'✓' if len(entities) >= 1 else '✗'}")
