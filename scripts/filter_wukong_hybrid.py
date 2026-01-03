#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合筛选方案：关键词快速预筛选 + Qwen 精确筛选
功能：
 1. 第一步：使用关键词快速过滤，减少需要 Qwen 处理的数据量
 2. 第二步：对预筛选结果使用 Qwen 模型精确判断
 3. 提高效率，节省计算资源

用法示例：
python scripts/filter_wukong_hybrid.py \
  --input_dir /mnt/disk/lxh/gill_data/wukong_release/wukong_release \
  --output_csv /mnt/disk/lxh/gill_data/wukong_filtered_spatial.csv \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --target_samples 20000
"""

import os
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List

# 扩展的中文关键词库（提升召回率，宁可多选不可漏选）
# 针对中文场景优化，包含介词组合、动词方位、数量词等

# 强方位词（明确的空间位置）
STRONG_KEYWORDS = [
    # 基础方位
    "左", "右", "上", "下", "中间", "中央", "顶部", "底部",
    "左上", "左下", "右上", "右下", "上方", "下方", "左侧", "右侧",
    "正中央", "正中间", "最上方", "最下方", "最左边", "最右边",
    
    # 介词组合（提升召回率）
    "在左边", "在右侧", "在下方", "在上方", "在中间", "在中央",
    "位于左侧", "位于右侧", "位于上方", "位于下方", "位于中间",
    "放在左边", "放在右边", "放在上方", "放在下方",
    "置于左侧", "置于右侧", "置于中央",
    
    # 动词方位（隐含位置关系）
    "坐于", "位于", "置于", "放在", "摆在", "挂在", "贴在",
    "排列在", "分布在", "分散在", "集中在",
    
    # 数量词组合（多物体通常隐含位置关系）
    "两个", "三个", "四个", "多个", "一对", "两对", "一组", "两组",
    "两只", "三只", "四只", "几个", "数个",
    
    # 相对位置描述
    "之间", "之中", "之内", "之外", "之前", "之后",
    "相对", "相对位置", "相对关系",
    
    # 布局描述词
    "布局", "排列", "分布", "排列方式", "空间布局",
    "横向", "纵向", "水平", "垂直", "对称", "不对称"
]

# 弱方位词（模糊的空间关系）
WEAK_KEYWORDS = [
    # 模糊方位
    "旁边", "周围", "四周", "环绕", "对角", "侧面",
    "背景", "前景", "附近", "周边", "邻近",
    
    # 环境描述
    "环境", "场景", "周围环境", "背景中", "前景中",
    "周围有", "附近有", "周边有",
    
    # 分布描述
    "分散", "集中", "聚集", "围绕", "包围",
    "零星", "密集", "稀疏",
    
    # 相对关系（弱）
    "相对", "相对于", "对比", "对比于"
]

# 直接实现，避免导入问题
def load_model(model_name: str, device: str = "cuda"):
    """加载 Qwen 文本模型（优先使用本地模型）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import os
    
    # 检查是否是本地路径（支持大小写不敏感匹配）
    is_local = False
    original_model_name = model_name
    if os.path.exists(model_name) and os.path.isdir(model_name):
        is_local = True
    else:
        # 尝试大小写不敏感匹配（如果路径是绝对路径且包含 model 目录）
        if os.path.isabs(model_name) and 'model' in model_name.lower():
            parent_dir = os.path.dirname(model_name)
            model_basename = os.path.basename(model_name)
            if os.path.exists(parent_dir):
                # 在父目录中查找大小写不匹配的目录
                for item in os.listdir(parent_dir):
                    if item.lower() == model_basename.lower() and os.path.isdir(os.path.join(parent_dir, item)):
                        model_name = os.path.join(parent_dir, item)
                        is_local = True
                        print(f"  🔍 检测到大小写不匹配，自动修正路径: {model_name}")
                        break
    
    if is_local:
        print(f"📦 从本地路径加载模型: {model_name}")
        # 本地模型：禁用网络请求
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,  # 只使用本地文件
            padding_side='left'  # decoder-only模型使用left padding
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True  # 只使用本地文件
        )
    else:
        print(f"📦 从 HuggingFace 加载模型: {model_name}")
        # 远程模型：临时禁用代理（如果代理服务未运行）
        old_proxy = os.environ.get('HTTP_PROXY')
        old_https_proxy = os.environ.get('HTTPS_PROXY')
        old_http_proxy = os.environ.get('http_proxy')
        old_https_proxy_lower = os.environ.get('https_proxy')
        try:
            # 临时禁用所有代理环境变量（避免代理连接失败）
            if old_proxy:
                os.environ.pop('HTTP_PROXY', None)
            if old_https_proxy:
                os.environ.pop('HTTPS_PROXY', None)
            if old_http_proxy:
                os.environ.pop('http_proxy', None)
            if old_https_proxy_lower:
                os.environ.pop('https_proxy', None)
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side='left'  # decoder-only模型使用left padding
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True
            )
        finally:
            # 恢复代理设置
            if old_proxy:
                os.environ['HTTP_PROXY'] = old_proxy
            if old_https_proxy:
                os.environ['HTTPS_PROXY'] = old_https_proxy
            if old_http_proxy:
                os.environ['http_proxy'] = old_http_proxy
            if old_https_proxy_lower:
                os.environ['https_proxy'] = old_https_proxy_lower
    
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print("✅ 模型加载完成")
    return model, tokenizer

def check_caption_with_qwen(caption: str, model, tokenizer, device: str = "cuda") -> tuple:
    """
    使用 Qwen 模型判断 caption（论文级 CoT Prompt）
    
    Returns:
        (type, reason): type 为 "strong"/"weak"/"none", reason 为判断理由
    """
    if not isinstance(caption, str) or len(caption.strip()) < 3:
        return (None, None)
    
    # 论文级 Prompt（Chain-of-Thought，视觉导向）
    prompt = f"""作为一个视觉数据集专家，请判断以下图像描述是否包含**具体的、可视觉化的**物体空间关系。

描述：{caption}

判别标准（请逐步思考）：
1. **实体要求**：必须包含至少两个具体的物理实体（如：人、物体、动物等），而不是抽象概念。
2. **位置要求**：必须包含明确的相对位置描述（如：左边、右边、上方、下方、中间、周围等）。
3. **排除规则**：
   - 抽象概念（如"社会底层"、"左翼思想"、"心底"）→ 判为 none
   - 单一物体居中（如"中间是一朵花"）→ 判为 strong（有效）
   - 无空间关系的并列描述（如"有猫和狗"）→ 判为 none
4. **分类规则**：
   - 明确方位词（左、右、上、下、中间、顶部、底部、左上、右下等）→ strong
   - 模糊方位词（旁边、周围、四周、环绕、背景、前景等）→ weak
   - 无位置信息或不符合上述要求 → none

请以 JSON 格式输出：
{{"type": "strong/weak/none", "reason": "简短判断理由（1-2句话）"}}"""
    
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            # 使用 generate 的参数，明确禁用采样相关参数以避免警告
            generation_config = model.generation_config if hasattr(model, 'generation_config') else None
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # 增加以容纳 JSON 和 reason
                do_sample=False,  # 使用贪心解码
                temperature=None,  # 明确设置为 None
                top_p=None,  # 明确设置为 None
                top_k=None,  # 明确设置为 None
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # 解析 JSON 响应
        import json
        import re
        
        # 尝试提取 JSON
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                result_type = result.get('type', '').lower()
                reason = result.get('reason', '')
                
                if result_type in ['strong', 'weak', 'none']:
                    return (result_type, reason)
            except:
                pass
        
        # 回退到简单匹配
        response_lower = response.lower()
        if "strong" in response_lower:
            return ("strong", "包含明确方位词")
        elif "weak" in response_lower:
            return ("weak", "包含模糊方位词")
        else:
            return ("none", "无空间关系")
            
    except Exception as e:
        # 如果模型调用失败，回退到关键词匹配
        caption_lower = caption.lower() if isinstance(caption, str) else ""
        has_strong = any(k in caption_lower for k in STRONG_KEYWORDS) if caption_lower else False
        has_weak = any(k in caption_lower for k in WEAK_KEYWORDS) if caption_lower else False
        
        if has_strong:
            return ("strong", "模型调用失败，使用关键词匹配")
        elif has_weak:
            return ("weak", "模型调用失败，使用关键词匹配")
        else:
            return ("none", "模型调用失败，无匹配关键词")

def fallback_keyword_check(caption: str) -> Optional[str]:
    """关键词匹配的备用方案"""
    if not isinstance(caption, str):
        return None
    
    caption_lower = caption.lower()
    
    has_strong = any(k in caption_lower for k in STRONG_KEYWORDS)
    has_weak = any(k in caption_lower for k in WEAK_KEYWORDS)
    
    if has_strong:
        return "strong"
    elif has_weak:
        return "weak"
    return None

def process_batch_truly(captions: List[str], model, tokenizer, device: str = "cuda", batch_size: int = 32) -> List[tuple]:
    """
    真正的批处理：一次处理多个caption，充分利用GPU
    4090 24GB显存可以支持更大的batch_size
    """
    import json
    import re
    
    if not captions:
        return []
    
    results = []
    total = len(captions)
    
    # 构建批处理prompt
    prompt_template = """作为一个视觉数据集专家，请判断以下图像描述是否包含**具体的、可视觉化的**物体空间关系。

判别标准（请逐步思考）：
1. **实体要求**：必须包含至少两个具体的物理实体（如：人、物体、动物等），而不是抽象概念。
2. **位置要求**：必须包含明确的相对位置描述（如：左边、右边、上方、下方、中间、周围等）。
3. **排除规则**：
   - 抽象概念（如"社会底层"、"左翼思想"、"心底"）→ 判为 none
   - 单一物体居中（如"中间是一朵花"）→ 判为 strong（有效）
   - 无空间关系的并列描述（如"有猫和狗"）→ 判为 none
4. **分类规则**：
   - 明确方位词（左、右、上、下、中间、顶部、底部、左上、右下等）→ strong
   - 模糊方位词（旁边、周围、四周、环绕、背景、前景等）→ weak
   - 无位置信息或不符合上述要求 → none

请以 JSON 格式输出：
{{"type": "strong/weak/none", "reason": "简短判断理由（1-2句话）"}}"""
    
    # 分批处理
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_captions = captions[batch_start:batch_end]
        
        # 为每个caption构建完整的prompt
        batch_prompts = []
        for caption in batch_captions:
            if not isinstance(caption, str) or len(caption.strip()) < 3:
                batch_prompts.append(None)
                continue
            full_prompt = f"""{prompt_template}

描述：{caption}"""
            batch_prompts.append(full_prompt)
        
        # 过滤掉None
        valid_indices = [i for i, p in enumerate(batch_prompts) if p is not None]
        if not valid_indices:
            # 如果整个批次都无效，返回None结果
            results.extend([(None, None)] * len(batch_captions))
            continue
        
        valid_prompts = [batch_prompts[i] for i in valid_indices]
        
        try:
            # 构建批处理消息
            messages_list = [[{"role": "user", "content": prompt}] for prompt in valid_prompts]
            texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) 
                     for msgs in messages_list]
            
            # 批处理tokenize
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            
            # 解码输出
            generated_ids = [
                out[len(inp):] for inp, out in zip(inputs['input_ids'], outputs)
            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 解析每个响应
            batch_results = [None] * len(batch_captions)
            for idx, (valid_idx, response) in enumerate(zip(valid_indices, responses)):
                response = response.strip()
                
                # 尝试解析JSON
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        result_type = result.get('type', '').lower()
                        reason = result.get('reason', '')
                        
                        if result_type in ['strong', 'weak', 'none']:
                            batch_results[valid_idx] = (result_type, reason)
                            continue
                    except:
                        pass
                
                # 回退到简单匹配
                response_lower = response.lower()
                if "strong" in response_lower:
                    batch_results[valid_idx] = ("strong", "包含明确方位词")
                elif "weak" in response_lower:
                    batch_results[valid_idx] = ("weak", "包含模糊方位词")
                else:
                    batch_results[valid_idx] = ("none", "无空间关系")
            
            # 处理无效的caption（返回None）
            for i in range(len(batch_captions)):
                if batch_results[i] is None:
                    batch_results[i] = (None, None)
            
            results.extend(batch_results)
            
            # 清理显存
            del inputs, outputs, generated_ids, responses
            torch.cuda.empty_cache()
            
        except Exception as e:
            # 如果批处理失败，回退到关键词匹配
            for caption in batch_captions:
                result = fallback_keyword_check(caption)
                if result:
                    results.append((result, "批处理失败，使用关键词匹配"))
                else:
                    results.append(("none", "批处理失败，无匹配关键词"))
        
        # 显示进度
        print(f"\r  🤖 Qwen 处理进度: {batch_end}/{total} ({100*batch_end/total:.1f}%)", end="", flush=True)
    
    print()  # 换行
    return results

def process_batch(captions: List[str], model, tokenizer, device: str = "cuda", batch_size: int = 32) -> List[tuple]:
    """批量处理，返回 (type, reason) 元组列表（使用真正的批处理）"""
    return process_batch_truly(captions, model, tokenizer, device, batch_size)

def quick_keyword_filter(caption: str) -> Optional[str]:
    """
    快速关键词预筛选
    返回: "candidate" (候选), None (不包含方位词)
    """
    if not isinstance(caption, str):
        return None
    
    caption_lower = caption.lower()
    
    has_strong = any(k in caption_lower for k in STRONG_KEYWORDS)
    has_weak = any(k in caption_lower for k in WEAK_KEYWORDS)
    
    if has_strong or has_weak:
        return "candidate"
    return None

def load_models_multi_gpu(model_name: str, num_gpus: int = 3):
    """在多GPU上加载模型"""
    models = []
    tokenizers = []
    devices = []
    
    # 检查是否为本地路径（支持大小写不敏感匹配）
    is_local = False
    if os.path.exists(model_name) and os.path.isdir(model_name):
        is_local = True
    else:
        # 尝试大小写不敏感匹配（如果路径是绝对路径且包含 model 目录）
        if os.path.isabs(model_name) and 'model' in model_name.lower():
            parent_dir = os.path.dirname(model_name)
            model_basename = os.path.basename(model_name)
            if os.path.exists(parent_dir):
                # 在父目录中查找大小写不匹配的目录
                for item in os.listdir(parent_dir):
                    if item.lower() == model_basename.lower() and os.path.isdir(os.path.join(parent_dir, item)):
                        model_name = os.path.join(parent_dir, item)
                        is_local = True
                        print(f"  🔍 检测到大小写不匹配，自动修正路径: {model_name}")
                        break
    
    print(f"🚀 在 {num_gpus} 张 GPU 上加载模型 {model_name} ...")
    if is_local:
        print(f"  📦 检测到本地模型路径，使用 local_files_only=True")
    else:
        print(f"  📦 从 HuggingFace 加载模型")
    
    for i in range(num_gpus):
        device = f"cuda:{i}"
        print(f"  📦 加载到 {device} ...")
        
        if is_local:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
                padding_side='left'  # decoder-only模型使用left padding
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            # 临时禁用代理（如果代理服务未运行）
            old_proxy = os.environ.get('HTTP_PROXY')
            old_https_proxy = os.environ.get('HTTPS_PROXY')
            old_http_proxy = os.environ.get('http_proxy')
            old_https_proxy_lower = os.environ.get('https_proxy')
            try:
                if old_proxy:
                    os.environ.pop('HTTP_PROXY', None)
                if old_https_proxy:
                    os.environ.pop('HTTPS_PROXY', None)
                if old_http_proxy:
                    os.environ.pop('http_proxy', None)
                if old_https_proxy_lower:
                    os.environ.pop('https_proxy', None)
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    padding_side='left'  # decoder-only模型使用left padding
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
            finally:
                if old_proxy:
                    os.environ['HTTP_PROXY'] = old_proxy
                if old_https_proxy:
                    os.environ['HTTPS_PROXY'] = old_https_proxy
                if old_http_proxy:
                    os.environ['http_proxy'] = old_http_proxy
                if old_https_proxy_lower:
                    os.environ['https_proxy'] = old_https_proxy_lower
        
        model.eval()
        models.append(model)
        tokenizers.append(tokenizer)
        devices.append(device)
        print(f"  ✅ {device} 加载完成")
    
    return models, tokenizers, devices

def main(args):
    # 多GPU支持
    if args.num_gpus > 1:
        models, tokenizers, devices = load_models_multi_gpu(args.model, args.num_gpus)
        current_gpu = 0  # 轮询使用GPU
    else:
        model, tokenizer = load_model(args.model, args.device)
        models = [model]
        tokenizers = [tokenizer]
        devices = [args.device]
    
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    
    # 使用自然排序（数字排序）而不是字符串排序
    import re
    def natural_sort_key(s):
        """自然排序：将数字部分按数值大小排序，而不是字符串排序"""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
    csv_files = sorted([str(p) for p in Path(args.input_dir).rglob("*.csv")], key=natural_sort_key)
    print(f"📦 找到 {len(csv_files)} 个 CSV 文件")
    
    # 从指定文件开始处理
    if args.start_from:
        start_file = args.start_from
        # 支持文件名或完整路径
        if not os.path.isabs(start_file):
            # 如果是文件名，查找匹配的文件
            start_idx = None
            for i, csv_file in enumerate(csv_files):
                if os.path.basename(csv_file) == start_file or start_file in csv_file:
                    start_idx = i
                    break
            if start_idx is not None:
                csv_files = csv_files[start_idx:]
                print(f"📍 从 {os.path.basename(csv_files[0])} 开始处理（跳过前 {start_idx} 个文件）")
            else:
                print(f"⚠️  警告：未找到文件 '{start_file}'，将从第一个文件开始处理")
        else:
            # 如果是完整路径，直接查找
            if start_file in csv_files:
                start_idx = csv_files.index(start_file)
                csv_files = csv_files[start_idx:]
                print(f"📍 从 {os.path.basename(csv_files[0])} 开始处理（跳过前 {start_idx} 个文件）")
            else:
                print(f"⚠️  警告：未找到文件 '{start_file}'，将从第一个文件开始处理")
    
    if args.limit_csvs:
        csv_files = csv_files[:args.limit_csvs]
    
    filtered_data = []
    total_processed = 0
    keyword_candidates = 0
    strong_count = 0
    weak_count = 0
    negative_count = 0  # 负样本计数
    
    processed_urls = set()
    initial_filtered_count = 0  # 记录已有输出文件中的记录数
    if os.path.exists(args.output_csv):
        try:
            existing_df = pd.read_csv(args.output_csv)
            processed_urls = set(existing_df['url'].astype(str))
            initial_filtered_count = len(existing_df)
            print(f"📂 检测到已有输出，已处理 {len(processed_urls)} 条（已有 {initial_filtered_count} 条记录）")
        except Exception:
            pass
    
    for csv_file in csv_files:
        # 修复：检查总处理数（包括已有记录），而不是缓冲区大小
        if args.target_samples:
            current_total = initial_filtered_count + total_processed
            if current_total >= args.target_samples:
                print(f"  ✅ 已达到目标样本数 {args.target_samples}（当前: {current_total}），停止处理")
                break
        
        print(f"\n🔍 处理 {os.path.basename(csv_file)}...")
        
        try:
            chunk_iter = pd.read_csv(csv_file, on_bad_lines='skip', chunksize=2000)
            csv_processed_count = 0
            csv_total_count = 0
            csv_skipped_chunks = 0  # 记录跳过的 chunk 数量
            csv_total_chunks = 0  # 记录总 chunk 数量
            
            for chunk in chunk_iter:
                csv_total_chunks += 1
                # 列名适配
                if 'url' not in chunk.columns and len(chunk.columns) >= 2:
                    chunk.rename(columns={chunk.columns[0]: 'url', chunk.columns[1]: 'caption'}, inplace=True)
                if 'text' in chunk.columns:
                    chunk.rename(columns={'text': 'caption'}, inplace=True)
                
                if 'caption' not in chunk.columns or 'url' not in chunk.columns:
                    continue
                
                csv_total_count += len(chunk)
                
                # 快速检查：如果所有 URL 都已处理，直接跳过（避免不必要的处理）
                chunk_urls = set(chunk['url'].astype(str))
                already_processed = chunk_urls & processed_urls
                if len(already_processed) == len(chunk_urls):
                    # 整个 chunk 都已处理，直接跳过
                    csv_processed_count += len(chunk)
                    csv_skipped_chunks += 1
                    continue
                
                # 过滤已处理
                chunk = chunk[~chunk['url'].astype(str).isin(processed_urls)]
                if len(chunk) == 0:
                    csv_processed_count += len(chunk_urls) - len(chunk)
                    csv_skipped_chunks += 1
                    continue
                
                # 优化：如果已跳过条数超过40，跳过整个chunk（避免处理大部分已处理的chunk）
                if len(already_processed) > 40:
                    csv_processed_count += len(chunk_urls)
                    csv_skipped_chunks += 1
                    print(f"  ⏭️  跳过整个chunk：已跳过 {len(already_processed)} 条（超过40条阈值）")
                    continue
                
                # 第一步：关键词快速预筛选
                candidates = []
                for idx, row in chunk.iterrows():
                    if quick_keyword_filter(row['caption']):
                        candidates.append((idx, row))
                
                keyword_candidates += len(candidates)
                print(f"  📋 关键词预筛选: {len(candidates)}/{len(chunk)} 候选 (已跳过 {len(already_processed)} 条)")
                
                if len(candidates) == 0:
                    continue
                
                # 第二步：Qwen 精确筛选
                print(f"  🤖 开始 Qwen 精确筛选 {len(candidates)} 个候选...")
                candidate_captions = [row['caption'] for _, row in candidates]
                
                # 选择GPU（多GPU时轮询）
                if args.num_gpus > 1:
                    gpu_idx = current_gpu % args.num_gpus
                    current_gpu += 1
                    model = models[gpu_idx]
                    tokenizer = tokenizers[gpu_idx]
                    device = devices[gpu_idx]
                else:
                    model = models[0]
                    tokenizer = tokenizers[0]
                    device = devices[0]
                
                results = process_batch(
                    candidate_captions,
                    model,
                    tokenizer,
                    device,
                    batch_size=args.batch_size
                )
                
                # 收集有效结果（包括负样本挖掘）
                negative_candidates = []  # 用于负样本挖掘
                
                for (idx, row), (result_type, reason) in zip(candidates, results):
                    if result_type in ["strong", "weak"]:
                        # 正样本：包含空间关系
                        filtered_data.append({
                            'url': row['url'],
                            'caption': row['caption'],
                            'spatial_type': result_type,
                            'reason': reason
                        })
                        
                        if result_type == "strong":
                            strong_count += 1
                        else:
                            weak_count += 1
                        
                        processed_urls.add(str(row['url']))
                        total_processed += 1
                    elif result_type == "none":
                        # 负样本候选：关键词召回但 LLM 判定为 none（伪空间关系）
                        negative_candidates.append((idx, row, reason))
                
                # 负样本挖掘：保留 10% 的 none 样本（用于对比学习）
                if negative_candidates and args.negative_ratio > 0:
                    import random
                    num_negative = max(1, int(len(negative_candidates) * args.negative_ratio))
                    selected_negative = random.sample(negative_candidates, min(num_negative, len(negative_candidates)))
                    
                    for idx, row, reason in selected_negative:
                        filtered_data.append({
                            'url': row['url'],
                            'caption': row['caption'],
                            'spatial_type': 'negative',  # 标记为负样本
                            'reason': reason
                        })
                        negative_count += 1
                        processed_urls.add(str(row['url']))
                        total_processed += 1
                
                # 更新 CSV 处理统计
                csv_processed_count += len(results)
                
                # 定期保存（每累积 50 条就保存，更频繁的保存）
                if len(filtered_data) >= 50:  # 改为 >= 50，更频繁保存
                    df_temp = pd.DataFrame(filtered_data)
                    if os.path.exists(args.output_csv):
                        df_temp.to_csv(args.output_csv, mode='a', header=False, index=False)
                    else:
                        df_temp.to_csv(args.output_csv, index=False)
                    filtered_data = []  # 清空缓冲区
                    current_total = initial_filtered_count + total_processed
                    print(f"  💾 已保存 {current_total} 条 (本次新增: {total_processed}, Strong: {strong_count}, Weak: {weak_count}, Negative: {negative_count})", flush=True)
                
                # 修复：检查总处理数（包括已有记录）
                if args.target_samples:
                    current_total = initial_filtered_count + total_processed
                    if current_total >= args.target_samples:
                        print(f"  ✅ 已达到目标样本数 {args.target_samples}（当前: {current_total}），停止处理")
                        break
            
            # 显示该 CSV 文件的处理统计
            if csv_total_count > 0:
                skip_ratio = csv_processed_count / csv_total_count * 100 if csv_total_count > 0 else 0
                if skip_ratio > 50:
                    print(f"  ⏭️  {os.path.basename(csv_file)}: 已跳过 {csv_processed_count}/{csv_total_count} 条 ({skip_ratio:.1f}%)")
            elif csv_total_chunks > 0 and csv_skipped_chunks == csv_total_chunks:
                # 如果所有 chunk 都被跳过，显示提示信息
                print(f"  ⏭️  {os.path.basename(csv_file)}: 所有 {csv_total_chunks} 个 chunk 都已跳过（已处理或超过阈值）")
            
            # 修复：检查总处理数（包括已有记录）
            if args.target_samples:
                current_total = initial_filtered_count + total_processed
                if current_total >= args.target_samples:
                    print(f"  ✅ 已达到目标样本数 {args.target_samples}（当前: {current_total}），停止处理")
                    break
                
        except Exception as e:
            print(f"⚠️ 处理 {csv_file} 时出错: {e}")
            continue
    
    # 保存剩余数据
    if filtered_data:
        df_temp = pd.DataFrame(filtered_data)
        if os.path.exists(args.output_csv):
            df_temp.to_csv(args.output_csv, mode='a', header=False, index=False)
        else:
            df_temp.to_csv(args.output_csv, index=False)
    
    # 最终统计和平衡
    if os.path.exists(args.output_csv):
        final_df = pd.read_csv(args.output_csv)
        final_strong = len(final_df[final_df['spatial_type'] == 'strong'])
        final_weak = len(final_df[final_df['spatial_type'] == 'weak'])
        final_negative = len(final_df[final_df['spatial_type'] == 'negative'])
        
        print(f"\n✅ 筛选完成！")
        print(f"📊 统计:")
        print(f"   关键词候选: {keyword_candidates}")
        print(f"   最终保留: {len(final_df)} 条")
        print(f"   Strong (强方位): {final_strong} ({final_strong/len(final_df):.1%})")
        print(f"   Weak (弱方位): {final_weak} ({final_weak/len(final_df):.1%})")
        if final_negative > 0:
            print(f"   Negative (负样本): {final_negative} ({final_negative/len(final_df):.1%})")
        print(f"   筛选率: {len(final_df)/keyword_candidates:.1%}" if keyword_candidates > 0 else "")
        
        # 如果需要调整比例，进行平衡采样
        if args.strong_ratio and final_strong > 0 and final_weak > 0:
            positive_df = final_df[final_df['spatial_type'].isin(['strong', 'weak'])]
            negative_df = final_df[final_df['spatial_type'] == 'negative']
            
            # 计算目标数量（基于正样本总数）
            total_positive = len(positive_df)
            target_strong = int(total_positive * args.strong_ratio)
            target_weak = total_positive - target_strong
            
            # 下采样
            strong_df = positive_df[positive_df['spatial_type'] == 'strong']
            weak_df = positive_df[positive_df['spatial_type'] == 'weak']
            
            # 检查 strong 样本是否足够
            if len(strong_df) < target_strong:
                print(f"\n⚠️  警告：Strong 样本不足！")
                print(f"   需要: {target_strong} 个，实际只有: {len(strong_df)} 个")
                print(f"   建议：筛选更多原始数据以获得足够的 strong 样本")
                print(f"   当前将使用所有 {len(strong_df)} 个 strong 样本")
                # 调整 weak 样本数量以匹配实际 strong 数量
                actual_strong_ratio = len(strong_df) / total_positive if total_positive > 0 else 0
                target_weak = total_positive - len(strong_df)
            else:
                strong_df = strong_df.sample(n=target_strong, random_state=42)
            
            if len(weak_df) > target_weak:
                weak_df = weak_df.sample(n=target_weak, random_state=42)
            
            # 处理 negative 样本（如果设置了 negative_ratio）
            if args.negative_ratio > 0 and len(negative_df) > 0:
                total_balanced = len(strong_df) + len(weak_df)
                target_negative = int(total_balanced * args.negative_ratio / (1 - args.negative_ratio))
                if len(negative_df) > target_negative:
                    negative_df = negative_df.sample(n=target_negative, random_state=42)
            
            # 合并并打乱
            balanced_df = pd.concat([strong_df, weak_df, negative_df]).sample(frac=1, random_state=42).reset_index(drop=True)
            balanced_path = args.output_csv.replace('.csv', '_balanced.csv')
            balanced_df.to_csv(balanced_path, index=False)
            
            print(f"\n📊 已生成平衡数据集:")
            print(f"   Strong: {len(strong_df)} ({len(strong_df)/len(balanced_df):.1%})")
            print(f"   Weak: {len(weak_df)} ({len(weak_df)/len(balanced_df):.1%})")
            if len(negative_df) > 0:
                print(f"   Negative: {len(negative_df)} ({len(negative_df)/len(balanced_df):.1%})")
            print(f"💾 保存到: {balanced_path}")
        
        print(f"💾 原始数据保存到: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="混合筛选方案：关键词预筛选 + Qwen 精确筛选（论文级优化版）"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="原始 Wukong CSV 文件夹路径")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="输出筛选后的 CSV 路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Qwen 模型路径或 HuggingFace 模型名")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--target_samples", type=int, default=20000,
                       help="目标样本数（建议 20000-50000）")
    parser.add_argument("--limit_csvs", type=int, default=None,
                       help="限制处理的 CSV 文件数量（用于测试）")
    parser.add_argument("--start_from", type=str, default=None,
                       help="从指定的 CSV 文件开始处理（可以是文件名如 'wukong_100m_1.csv' 或完整路径）")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批处理大小（4090 24GB显存建议32-64，多GPU时可更大）")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="使用的GPU数量（默认1，支持多GPU并行，建议3张4090）")
    parser.add_argument("--strong_ratio", type=float, default=0.8,
                       help="Strong 样本的目标比例（0.8 表示 80%%）")
    parser.add_argument("--negative_ratio", type=float, default=0.1,
                       help="负样本比例（用于对比学习，0.1 表示 10%%）")
    args = parser.parse_args()
    main(args)

