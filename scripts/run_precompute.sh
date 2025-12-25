#!/bin/bash
# 预计算 CLIP Embeddings 启动脚本

# 激活 conda 环境
source /usr/miniconda3/etc/profile.d/conda.sh
conda activate gill_chinese

# 运行预计算脚本
cd /home/lxh/Project/gill-main

python scripts/precompute_wukong_embeddings.py \
  --image_dir /mnt/disk/lxh/gill_data/wukong_500k/images \
  --clip_model_path ./model/chinese-clip-vit-base-patch16 \
  --batch_size 64

echo "预计算完成！"

