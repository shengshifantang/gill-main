#!/bin/bash
# GILL 训练启动脚本 - 单 GPU 模式（备用方案）

# 激活 conda 环境
source /usr/miniconda3/etc/profile.d/conda.sh
conda activate gill_chinese

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 清理之前的运行目录
rm -rf /home/lxh/Project/gill-main/runs/test_run*

# 切换到项目目录
cd /home/lxh/Project/gill-main

# 指定使用单个 GPU
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "开始训练 GILL 模型 (单 GPU 模式)"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

python main.py \
  --dataset wukong \
  --val-dataset wukong \
  --dataset-dir /mnt/disk/lxh/gill_data/wukong_500k \
  --image-dir /mnt/disk/lxh/gill_data/wukong_500k/images \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --steps_per_epoch 100 \
  --epochs 1 \
  --opt-version ./model/qwen2.5-7B-Instruct \
  --visual-model ./model/chinese-clip-vit-base-patch16 \
  --exp-name test_run_single \
  --workers 4

echo "=========================================="
echo "训练完成！"
echo "=========================================="

