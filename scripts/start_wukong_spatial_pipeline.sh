#!/usr/bin/env bash

###############################################################################
# 一键启动：悟空数据 -> 过滤 -> 标注(Qwen2-VL-7B) -> 训练 Spatial Adapter
#
# 使用前请先根据你本机情况修改下面几项配置变量。
###############################################################################

set -euo pipefail

########################
# 路径与参数配置（必改）
########################

# 悟空图片根目录（上一阶段过滤出的 50w+ 图片就放在这里）
DATA_ROOT="/mnt/disk/lxh/gill_data"

# 含有 image_path + caption 的原始 JSONL（必须已有 caption）
# 每行格式示例：
# {"image_path": "xxx/yyy.jpg", "caption": "一只猫在左边，一只狗在右边"}
CAPTION_JSONL="/mnt/disk/lxh/gill_data/wukong_with_caption.jsonl"

# 中间文件与输出目录
WORK_DIR="/mnt/disk/lxh/Project/gill-data"
FILTER_INDEX="${DATA_ROOT}/filter_index.txt"
RAW_JSONL="${WORK_DIR}/wukong_raw.jsonl"          # 过滤 + 关联 caption 后的输入
LABELED_JSONL="${WORK_DIR}/wukong_labeled.jsonl"  # Qwen2-VL 标注后的输出

# Qwen2-VL 模型路径或名称（可用本地路径或 HF 名称）
# 集群无法访问外网时，请使用本地已下载的模型路径
QWEN_VL_MODEL="/mnt/disk/lxh/models/Qwen2-VL-2B-Instruct"

# 训练相关
KOLORS_MODEL="./model/Kolors"
ADAPTER_OUT="./checkpoints/spatial_adapter_wukong"
BATCH_SIZE=4
EPOCHS=10
LR=1e-4

# 过滤参数
MIN_SIZE=256
WORKERS=8

########################
# 0. 环境检查
########################

echo ">>> 使用配置："
echo "DATA_ROOT      = ${DATA_ROOT}"
echo "CAPTION_JSONL  = ${CAPTION_JSONL}"
echo "WORK_DIR       = ${WORK_DIR}"
echo "QWEN_VL_MODEL  = ${QWEN_VL_MODEL}"
echo "KOLORS_MODEL   = ${KOLORS_MODEL}"
echo

if [ ! -d "${DATA_ROOT}" ]; then
  echo "错误：DATA_ROOT 目录不存在：${DATA_ROOT}" >&2
  exit 1
fi

if [ ! -f "${CAPTION_JSONL}" ]; then
  echo "错误：找不到 CAPTION_JSONL：${CAPTION_JSONL}" >&2
  echo "请准备包含 image_path + caption 字段的 JSONL 再运行本脚本。" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"

########################
# 1. 过滤小图，生成 filter_index.txt
########################

echo ">>> [1/4] 过滤小图，生成 ${FILTER_INDEX} ..."
python scripts/filter_wukong_images.py \
  --root "${DATA_ROOT}" \
  --out "${FILTER_INDEX}" \
  --min-size ${MIN_SIZE} \
  --workers ${WORKERS}

########################
# 2. 结合 caption，生成 wukong_raw.jsonl
########################

echo ">>> [2/4] 结合 caption，生成 ${RAW_JSONL} ..."
python - <<PY
import json, os

caption_path = "${CAPTION_JSONL}"
filter_index = "${FILTER_INDEX}"
out_path = "${RAW_JSONL}"

print(f"读取过滤索引: {filter_index}")
keep = set()
with open(filter_index, "r", encoding="utf-8") as f:
    for line in f:
        rel = line.strip()
        if rel:
            keep.add(rel)

print(f"过滤后图片数量: {len(keep)}")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
cnt_in, cnt_out = 0, 0

with open(caption_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        cnt_in += 1
        try:
            item = json.loads(line)
        except Exception:
            continue
        rel = item.get("image_path") or item.get("image")
        if not rel:
            continue
        # 若 caption 文件里的路径是绝对路径，则换算为相对 DATA_ROOT 的形式
        if os.path.isabs(rel):
            try:
                rel = os.path.relpath(rel, "${DATA_ROOT}")
            except ValueError:
                # 不在 DATA_ROOT 下的，直接跳过
                continue
        if rel not in keep:
            continue
        # 只保留必要字段
        out_item = {
            "image_path": rel,
            "caption": item.get("caption", "")
        }
        fout.write(json.dumps(out_item, ensure_ascii=False) + "\\n")
        cnt_out += 1

print(f"原始 caption 行数: {cnt_in}, 过滤后写入: {cnt_out}")
PY

########################
# 3. 运行 Qwen2-VL 标注（可先小批验证，再全量）
########################

echo ">>> [3/4] 运行 Qwen2-VL 标注（建议先小批验证 500 条）"

echo ">>> 3.1 小批验证（500 条）..."
python scripts/annotate_wukong_qwen2vl.py \
  --input "${RAW_JSONL}" \
  --image-root "${DATA_ROOT}" \
  --output "${LABELED_JSONL}" \
  --model "${QWEN_VL_MODEL}" \
  --device cuda \
  --batch-size 4 \
  --max-samples 500

echo ">>> 小批验证已完成，请人工抽查 ${LABELED_JSONL} 前几行与若干可视化样例。"
echo "确认质量后，按回车继续全量标注（或 Ctrl+C 终止）..."
read -r _

echo ">>> 3.2 全量标注（断点续传，输出追加写入）..."
python scripts/annotate_wukong_qwen2vl.py \
  --input "${RAW_JSONL}" \
  --image-root "${DATA_ROOT}" \
  --output "${LABELED_JSONL}" \
  --model "${QWEN_VL_MODEL}" \
  --device cuda \
  --batch-size 4

########################
# 4. 使用标注数据训练 Spatial Adapter
########################

echo ">>> [4/4] 使用标注数据训练 Spatial Adapter ..."
python scripts/train_spatial_adapter.py \
  --mixed-data "${LABELED_JSONL}" \
  --kolors-model "${KOLORS_MODEL}" \
  --output-dir "${ADAPTER_OUT}" \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --image-dir "${DATA_ROOT}"

echo "✅ 全流程完成。"
echo "标注数据: ${LABELED_JSONL}"
echo "Adapter 权重: ${ADAPTER_OUT}"


