"""预计算 Wukong 数据集的 CLIP Image Embeddings

这个脚本会为 Wukong 数据集中的所有图片预先计算 CLIP embeddings，
并保存为 .npy 文件，供 GILL 训练时使用。
"""
import os
import argparse
import glob
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModel


def main(args):
    print(f"正在加载 CLIP 模型: {args.clip_model_path}...")
    processor = AutoProcessor.from_pretrained(args.clip_model_path)
    model = AutoModel.from_pretrained(args.clip_model_path).cuda().eval()

    # 确保输出目录存在
    output_dir = os.path.join(args.image_dir, "clip_embs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片
    # 支持 jpg, jpeg, png
    print(f"正在扫描图片目录: {args.image_dir}")
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(args.image_dir, "*.png")) + \
                  glob.glob(os.path.join(args.image_dir, "*.jpeg"))
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"Embeddings 将保存到: {output_dir}")
    
    # 检查已经处理过的文件
    existing_embs = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
    print(f"已存在 {len(existing_embs)} 个 embedding 文件")
    
    # 过滤掉已经处理过的图片
    if not args.force:
        image_files_to_process = []
        for f in image_files:
            filename = os.path.basename(f)
            emb_filename = f"{filename}.npy"
            if emb_filename not in existing_embs:
                image_files_to_process.append(f)
        print(f"需要处理 {len(image_files_to_process)} 张新图片 (使用 --force 强制重新处理所有图片)")
        image_files = image_files_to_process
    
    if len(image_files) == 0:
        print("没有需要处理的图片，退出")
        return
    
    batch_size = args.batch_size
    
    # 批量处理
    print(f"开始批量处理 (batch_size={batch_size})...")
    for i in tqdm(range(0, len(image_files), batch_size), desc="处理进度"):
        batch_files = image_files[i : i + batch_size]
        batch_images = []
        valid_indices = []
        
        # 加载图片
        for idx, f in enumerate(batch_files):
            try:
                img = Image.open(f).convert('RGB')
                batch_images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"\n读取图片出错 {f}: {e}")
        
        if not batch_images:
            continue
            
        # 推理
        try:
            inputs = processor(images=batch_images, return_tensors="pt").to("cuda")
            with torch.no_grad():
                vision_outputs = model.get_image_features(**inputs)
                # 归一化 (GILL 需要归一化的特征)
                vision_outputs = vision_outputs / vision_outputs.norm(dim=-1, keepdim=True)
                
            # 保存
            vision_outputs = vision_outputs.cpu().numpy()
            for j, valid_idx in enumerate(valid_indices):
                filename = os.path.basename(batch_files[valid_idx])
                save_path = os.path.join(output_dir, f"{filename}.npy")
                np.save(save_path, vision_outputs[j])
        except Exception as e:
            print(f"\n处理批次出错: {e}")
            # 如果批次处理失败，尝试单个处理
            print("尝试单个处理...")
            for idx, f in enumerate(batch_files):
                try:
                    img = Image.open(f).convert('RGB')
                    inputs = processor(images=[img], return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        vision_output = model.get_image_features(**inputs)
                        vision_output = vision_output / vision_output.norm(dim=-1, keepdim=True)
                    vision_output = vision_output.cpu().numpy()
                    filename = os.path.basename(f)
                    save_path = os.path.join(output_dir, f"{filename}.npy")
                    np.save(save_path, vision_output[0])
                except Exception as e2:
                    print(f"单个处理也失败 {f}: {e2}")
    
    print(f"\n完成！所�� embeddings 已保存到: {output_dir}")
    print(f"总共生成了 {len(os.listdir(output_dir))} 个 .npy 文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预计算 Wukong 数据集的 CLIP Embeddings")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="图片目录路径 (例如: /mnt/disk/lxh/gill_data/wukong_500k/images)")
    parser.add_argument("--clip_model_path", type=str, required=True, 
                        help="Chinese-CLIP 模型路径 (例如: ./model/chinese-clip-vit-base-patch16)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批处理大小 (默认: 64)")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理所有图片，即使已存在 embedding 文件")
    args = parser.parse_args()
    main(args)

