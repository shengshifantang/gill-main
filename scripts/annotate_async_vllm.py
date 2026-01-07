#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 vLLM 的异步高并发标注脚本（Qwen2.5-VL-32B）

架构：
- 服务端：vLLM API Server (Tensor Parallelism=3)
- 客户端：AsyncIO 异步并发请求

优势：
1. 架构解耦：服务端和客户端分离，更稳定
2. Continuous Batching：vLLM 自动优化批处理
3. 高并发：单线程可处理数千个并发请求
4. 断点续传：自动跳过已处理的数据

服务端启动命令（在独立终端运行）：
export CUDA_VISIBLE_DEVICES=0,1,2
python -m vllm.entrypoints.openai.api_server \
    --model /root/models/Qwen2.5-VL-32B-Instruct-AWQ \
    --quantization awq \
    --tensor-parallel-size 3 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --disable-log-requests

客户端使用示例：
python scripts/annotate_async_vllm.py \
    --input /mnt/disk/lxh/gill_data/wukong_downloaded_500k.jsonl \
    --image-root /mnt/disk/lxh/gill_data/wukong_images \
    --output /mnt/disk/lxh/gill_data/wukong_labeled_vllm.jsonl \
    --api-base http://localhost:8000/v1 \
    --model-name /root/models/Qwen2.5-VL-32B-Instruct-AWQ \
    --max-concurrency 32
"""

import os
import json
import asyncio
import base64
import argparse
import re
from typing import Set, Dict, Any, Optional
from pathlib import Path

try:
    from tqdm.asyncio import tqdm as async_tqdm
    HAS_ASYNC_TQDM = True
except ImportError:
    # 如果 tqdm 版本不支持 asyncio，使用普通 tqdm
    from tqdm import tqdm
    HAS_ASYNC_TQDM = False

try:
    from openai import AsyncOpenAI
except ImportError:
    print("❌ 需要安装 openai 库: pip install openai")
    exit(1)


# ================= 配置区域 =================
DEFAULT_MAX_CONCURRENCY = 32  # 建议根据显存负载调整：2x4090 TP=2 建议 32-50（降低并发数可减少超时错误）
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"  # vLLM 不需要真实的 API Key
# ===========================================


def encode_image_base64(image_path: str) -> str:
    """快速读取图片并转为Base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def build_prompt(caption: str) -> str:
    """
    构建 Reasoning-Aware Prompt（Few-shot 优化版）
    
    加入具体示例可以显著降低格式错误率，提升数据质量。
    """
    return f"""你是一个空间智能专家。请分析图片中与描述"{caption}"相关的实体。

请严格按照以下步骤思考：
1. **Rationale**: 运用空间推理，解释为什么物体位于该位置（考虑遮挡、支撑、透视关系）。
2. **Detection**: 输出严格的JSON格式。

示例输入：描述"一只猫坐在沙发上"
示例输出：
{{
    "rationale": "猫是画面主体，位于图像中心偏下；沙发作为支撑物体位于底部。",
    "objects": [
        {{"name": "猫", "bbox": [200, 300, 600, 700]}},
        {{"name": "沙发", "bbox": [100, 800, 900, 1000]}}
    ]
}}

当前任务描述："{caption}"

注意：
- 坐标请使用 0-1000 的归一化整数（相对于图片尺寸）
- bbox 格式：[x1, y1, x2, y2]，其中 (x1,y1) 是左上角，(x2,y2) 是右下角
- 确保 x1 < x2 且 y1 < y2
- 如果图片中没有描述中的物体，objects 列表为空
- 只输出 JSON，不要有其他文字或 Markdown 标记"""


def sanitize_bbox(bbox: list, width: int = 1000, height: int = 1000) -> Optional[list]:
    """
    [关键] 坐标清洗与验证
    
    1. 确保坐标是数字
    2. 自动检测 0-1 范围的归一化坐标并转换为 0-1000
    3. 确保 x1 < x2, y1 < y2
    4. 裁剪到 [0, 1000] 范围
    5. 过滤无效框（面积过小或反向框）
    
    这对于训练稳定性至关重要，防止 NaN/Inf 导致 Loss 异常。
    """
    try:
        # 强制转 float（处理字符串数字）
        b = [float(x) for x in bbox]
        if len(b) != 4:
            return None
        
        # [优化] 自动检测 0-1 范围的归一化坐标
        # 如果所有坐标都在 0.0-1.0 之间，则自动乘以 1000
        if all(0.0 <= x <= 1.0 for x in b):
            b = [x * 1000 for x in b]
        
        # 转换为整数
        b = [int(x) for x in b]
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        
        # 确保顺序正确（左上角 < 右下角）
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # 裁剪到有效范围
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        
        # 过滤无效框（面积过小或反向框）
        if x2 <= x1 + 10 or y2 <= y1 + 10:
            return None
        
        return [x1, y1, x2, y2]
    except (ValueError, TypeError, IndexError):
        return None


def robust_parse_json(content: str) -> Optional[Dict[str, Any]]:
    """
    [关键] 鲁棒的 JSON 提取器，应对 Markdown 和文本噪声
    
    改进点：
    1. 移除 Markdown 代码块标记
    2. 处理注释和多余文本
    3. 坐标清洗和验证
    4. 结构验证
    
    这能显著提升数据利用率，减少因格式问题导致的有效数据丢失。
    """
    if not content:
        return None
    
    # 1. 移除 Markdown 代码块标记
    content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'```', '', content)
    
    # 2. 移除可能的注释（// 或 # 开头的行）
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith('//') and not stripped.startswith('#'):
            cleaned_lines.append(line)
    content = '\n'.join(cleaned_lines)
    
    # 3. 寻找最外层大括号
    start = content.find('{')
    end = content.rfind('}')
    
    if start == -1 or end == -1 or start >= end:
        return None
    
    json_str = content[start:end+1]
    
    try:
        # 4. 标准 JSON 解析
        data = json.loads(json_str)
        
        # 5. 结构验证与清洗
        if not isinstance(data, dict):
            return None
        
        if "objects" in data and isinstance(data["objects"], list):
            valid_objs = []
            for obj in data["objects"]:
                if isinstance(obj, dict) and "name" in obj and "bbox" in obj:
                    if isinstance(obj["bbox"], list):
                        clean_bbox = sanitize_bbox(obj["bbox"])
                        if clean_bbox:
                            obj["bbox"] = clean_bbox
                            valid_objs.append(obj)
            data["objects"] = valid_objs
        
        # 验证至少包含 objects 字段
        if "objects" not in data:
            return None
        
        return data
        
    except json.JSONDecodeError as e:
        # JSON 解析失败，可能是格式问题
        return None
    except Exception:
        return None


def parse_bboxes_from_content(content: str) -> Optional[Dict[str, Any]]:
    """从模型输出中解析 JSON（使用鲁棒解析器）"""
    return robust_parse_json(content)


class AnnotationWorker:
    def __init__(self, args):
        self.args = args
        self.client = AsyncOpenAI(
            api_key=args.api_key or DEFAULT_API_KEY,
            base_url=args.api_base or DEFAULT_API_BASE
        )
        # 先初始化 stats，因为 _load_progress 会使用它
        self.stats = {
            'total': 0,
            'processed': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'parse_error': 0,  # JSON 解析失败
            'invalid_bbox': 0  # 坐标无效
        }
        self.processed_paths = self._load_progress()
        self.max_concurrency = args.max_concurrency
        # 注意：semaphore 和 lock 在 run() 方法中创建，确保在正确的事件循环中
        self.semaphore = None
        self.write_lock = None
        self.error_log_path = args.output.replace(".jsonl", "_errors.jsonl")  # 错误日志

    def _load_progress(self) -> Set[str]:
        """加载已处理的图片路径（标准化路径以确保一致性）"""
        processed = set()
        if os.path.exists(self.args.output):
            print(f"📖 读取断点续传文件: {self.args.output}")
            # 修复末尾可能缺失的换行符
            self._fix_newline(self.args.output)
            with open(self.args.output, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        image_path = data.get('image_path')
                        if image_path:
                            # 标准化路径（与 process_single_item 保持一致）
                            if not os.path.isabs(image_path):
                                full_path = os.path.join(self.args.image_root, image_path)
                            else:
                                full_path = image_path
                            normalized_path = os.path.normpath(full_path)
                            processed.add(normalized_path)
                    except:
                        pass
        print(f"✅ 已完成: {len(processed)} 条")
        self.stats['skipped'] = len(processed)
        return processed

    def _fix_newline(self, filepath: str):
        """确保文件以换行符结尾"""
        try:
            with open(filepath, 'rb+') as f:
                f.seek(0, 2)  # 移动到文件末尾
                if f.tell() > 0:  # 如果文件不为空
                    f.seek(-1, 2)  # 移动到倒数第一个字节
                    if f.read(1) != b'\n':
                        f.write(b'\n')
        except Exception:
            pass

    async def process_single_item(self, item: Dict[str, Any], pbar) -> None:
        """处理单张图片"""
        image_path = item.get('image_path', '')
        if not image_path:
            pbar.update(1)
            return

        # 处理相对路径
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.args.image_root, image_path)

        # 检查文件是否存在
        if not os.path.exists(image_path):
            self.stats['failed'] += 1
            pbar.update(1)
            return

        # 限制并发
        async with self.semaphore:
            try:
                # 准备请求
                b64_img = await asyncio.to_thread(encode_image_base64, image_path)
                prompt = build_prompt(item.get('caption', ''))

                # [优化] 添加超时控制（120秒），防止请求卡死
                try:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                    model=self.args.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                            ],
                        }
                    ],
                    max_tokens=512,
                    temperature=0.1,  # 低温度保证格式稳定
                    top_p=0.9,
                        ),
                        timeout=120.0  # 120秒超时（从60秒增加到120秒，减少超时错误）
                )
                except asyncio.TimeoutError:
                    raise Exception("Request timeout after 120 seconds")

                content = response.choices[0].message.content

                # 解析结果
                parsed_annotations = parse_bboxes_from_content(content)

                # 构建结果项
                result_item = item.copy()
                result_item['vlm_output'] = content  # 保存原始输出
                
                if parsed_annotations:
                    objects = parsed_annotations.get('objects', [])
                    if objects:
                        result_item['annotations'] = parsed_annotations
                        result_item['objects'] = objects
                        self.stats['success'] += 1
                    else:
                        # 解析成功但没有有效对象
                        result_item['annotations'] = parsed_annotations
                        result_item['objects'] = []
                        result_item['no_objects'] = True
                        self.stats['success'] += 1  # 仍然算成功（可能是图片中确实没有物体）
                else:
                    # JSON 解析失败
                    result_item['annotations_error'] = True
                    result_item['error_type'] = 'parse_error'
                    self.stats['parse_error'] += 1
                    self.stats['failed'] += 1

                # 异步写入结果（加锁保证线程安全）
                async with self.write_lock:
                    with open(self.args.output, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            except Exception as e:
                # 错误处理：记录失败但不中断流程
                self.stats['failed'] += 1
                
                # [关键] 记录错误日志（用于论文的 Failure Analysis）
                error_entry = {
                    "image_path": image_path,
                    "caption": item.get('caption', ''),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "raw_output": content if 'content' in locals() else ""
                }
                
                async with self.write_lock:
                    try:
                        with open(self.error_log_path, 'a', encoding='utf-8') as f_err:
                            f_err.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                    except Exception:
                        pass  # 错误日志写入失败不影响主流程
            finally:
                self.stats['processed'] += 1
                pbar.update(1)

    async def run(self):
        """主运行函数"""
        # 在事件循环中创建 Semaphore 和 Lock（修复事件循环问题）
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.write_lock = asyncio.Lock()
        
        # 1. 读取输入数据（跳过已处理的，使用标准化路径）
        tasks_data = []
        print(f"📖 读取输入文件: {self.args.input}")
        with open(self.args.input, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    image_path = item.get('image_path', '')
                    if image_path:
                        # 标准化路径（与 _load_progress 保持一致）
                        if not os.path.isabs(image_path):
                            full_path = os.path.join(self.args.image_root, image_path)
                        else:
                            full_path = image_path
                        normalized_path = os.path.normpath(full_path)
                        if normalized_path not in self.processed_paths:
                            tasks_data.append(item)
                except json.JSONDecodeError:
                    continue

        self.stats['total'] = len(tasks_data)
        print(f"🚀 待处理任务数: {len(tasks_data)}")

        if len(tasks_data) == 0:
            print("✅ 所有任务已完成，无需处理")
            return

        # 2. 创建进度条
        if HAS_ASYNC_TQDM:
            progress_bar = async_tqdm(
                total=len(tasks_data),
                desc="标注进度",
                unit="img",
                ncols=100
            )
        else:
            from tqdm import tqdm
            progress_bar = tqdm(
                total=len(tasks_data),
                desc="标注进度",
                unit="img",
                ncols=100
            )

        # 3. 创建任务列表
        tasks = [
            self.process_single_item(item, progress_bar)
            for item in tasks_data
        ]

        # 4. 并发执行
        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        progress_bar.close()

        # 5. 输出统计信息
        elapsed = end_time - start_time
        print(f"\n{'='*60}")
        print(f"📊 处理完成统计")
        print(f"{'='*60}")
        print(f"总任务数: {self.stats['total']}")
        print(f"成功: {self.stats['success']}")
        print(f"失败: {self.stats['failed']}")
        print(f"  - JSON 解析失败: {self.stats['parse_error']}")
        print(f"  - 其他错误: {self.stats['failed'] - self.stats['parse_error']}")
        print(f"跳过: {self.stats['skipped']}")
        print(f"耗时: {elapsed:.2f} 秒")
        if self.stats['processed'] > 0:
            print(f"平均速度: {self.stats['processed']/elapsed:.2f} 图片/秒")
        if self.stats['total'] > 0:
            success_rate = (self.stats['success'] / self.stats['total']) * 100
            print(f"成功率: {success_rate:.2f}%")
        print(f"{'='*60}")
        print(f"📝 错误日志已保存到: {self.error_log_path}")
        print(f"   可用于 Failure Analysis 和论文的 Limitations 章节")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="基于 vLLM 的异步高并发标注脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
1. 启动 vLLM 服务端（在独立终端）：
   export CUDA_VISIBLE_DEVICES=0,1,2
   python -m vllm.entrypoints.openai.api_server \\
       --model /root/models/Qwen2.5-VL-32B-Instruct-AWQ \\
       --quantization awq \\
       --tensor-parallel-size 3 \\
       --trust-remote-code \\
       --max-model-len 8192 \\
       --gpu-memory-utilization 0.95 \\
       --port 8000

2. 运行客户端脚本：
   python scripts/annotate_async_vllm.py \\
       --input wukong_downloaded.jsonl \\
       --image-root ./images \\
       --output wukong_labeled.jsonl \\
       --api-base http://localhost:8000/v1 \\
       --model-name /root/models/Qwen2.5-VL-32B-Instruct-AWQ \\
       --max-concurrency 32
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL 文件路径（包含 image_path 和 caption）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL 文件路径"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="图片根目录（用于解析相对路径）"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"vLLM API 服务地址（默认: {DEFAULT_API_BASE}）"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help=f"API Key（vLLM 不需要真实 key，默认: {DEFAULT_API_KEY}）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称（必须与 vLLM 启动参数中的 --model 一致）"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"最大并发数（默认: {DEFAULT_MAX_CONCURRENCY}，建议根据显存调整）"
    )

    args = parser.parse_args()

    # 验证输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 运行标注任务
    worker = AnnotationWorker(args)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()

