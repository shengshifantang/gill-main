#!/bin/bash
# 数据一致性检查脚本

INPUT_FILE="/mnt/disk/lxh/gill_data/wukong_downloaded_500k_fixed.jsonl"
OUTPUT_FILE="/mnt/disk/lxh/gill_data/wukong_labeled.jsonl"
ERROR_FILE="/mnt/disk/lxh/gill_data/wukong_labeled_errors.jsonl"
IMAGE_ROOT="/mnt/disk/lxh/gill_data/images"

echo "============================================================"
echo "🔍 数据一致性检查"
echo "============================================================"
echo ""

python3 << 'PYTHON_SCRIPT'
import json
import os

input_file = "/mnt/disk/lxh/gill_data/wukong_downloaded_500k_fixed.jsonl"
output_file = "/mnt/disk/lxh/gill_data/wukong_labeled.jsonl"
error_file = "/mnt/disk/lxh/gill_data/wukong_labeled_errors.jsonl"
image_root = "/mnt/disk/lxh/gill_data/images"

def normalize_path(path, image_root):
    """标准化路径"""
    if not os.path.isabs(path):
        return os.path.normpath(os.path.join(image_root, path))
    return os.path.normpath(path)

# 1. 检查输出文件中的重复
print("1️⃣ 检查输出文件中的重复数据...")
output_paths = {}
duplicates = []
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                image_path = data.get('image_path', '')
                if image_path:
                    normalized = normalize_path(image_path, image_root)
                    if normalized in output_paths:
                        duplicates.append((image_path, output_paths[normalized], line_num))
                    else:
                        output_paths[normalized] = line_num
            except:
                pass
    print(f"   总行数: {line_num}")
    print(f"   唯一 image_path: {len(output_paths)}")
    if duplicates:
        print(f"   ⚠️  发现 {len(duplicates)} 个重复")
        for img, first, dup in duplicates[:3]:
            print(f"      重复: {os.path.basename(img)} (行{first}, 行{dup})")
    else:
        print("   ✅ 无重复")

# 2. 检查输出文件中的数据质量
print("\n2️⃣ 检查输出文件中的数据质量...")
success_count = 0
error_count = 0
no_objects_count = 0
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'annotations_error' in data or 'error_type' in data:
                    error_count += 1
                elif data.get('no_objects', False):
                    no_objects_count += 1
                elif 'annotations' in data or 'objects' in data:
                    success_count += 1
            except:
                pass
    print(f"   成功标注: {success_count}")
    print(f"   标注错误: {error_count}")
    print(f"   无对象: {no_objects_count}")
    print(f"   总计: {success_count + error_count + no_objects_count}")

# 3. 检查错误日志和输出文件的重叠
print("\n3️⃣ 检查错误日志与输出文件的重叠...")
if os.path.exists(error_file) and os.path.exists(output_file):
    error_paths = set()
    with open(error_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                error_entry = json.loads(line.strip())
                image_path = error_entry.get('image_path', '')
                if image_path:
                    error_paths.add(normalize_path(image_path, image_root))
            except:
                pass
    
    output_paths_normalized = set(output_paths.keys())
    overlap = error_paths & output_paths_normalized
    
    print(f"   错误日志总数: {len(error_paths)}")
    print(f"   输出文件总数: {len(output_paths_normalized)}")
    print(f"   重叠数量: {len(overlap)}")
    if len(overlap) > 0:
        print(f"   ⚠️  有 {len(overlap)} 条数据既在错误日志中，又在输出文件中")
        print("      这些是重试成功的数据，会在下次运行重试脚本时自动清理")
    else:
        print("   ✅ 无重叠（错误日志已正确清理）")

# 4. 检查输入文件完整性
print("\n4️⃣ 检查输入文件完整性...")
if os.path.exists(input_file):
    input_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            input_count += 1
    
    input_in_output = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                img_path = item.get('image_path', '')
                if img_path:
                    normalized = normalize_path(img_path, image_root)
                    if normalized in output_paths_normalized:
                        input_in_output += 1
            except:
                pass
    
    print(f"   输入文件总数: {input_count}")
    print(f"   输入文件中已处理: {input_in_output}")
    print(f"   输入文件中未处理: {input_count - input_in_output}")
    if input_count > 0:
        progress = (input_in_output / input_count) * 100
        print(f"   处理进度: {progress:.2f}%")

print("\n" + "=" * 60)
print("✅ 检查完成")
print("=" * 60)
PYTHON_SCRIPT
