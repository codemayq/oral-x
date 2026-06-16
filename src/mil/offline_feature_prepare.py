import os
import glob
import pandas as pd
import torch
import openslide
import numpy as np
import timm
from tqdm import tqdm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from .config import Config
# 导入修改后的函数和模型加载器
from .wsi_processor import extract_wsi_features, get_virchow2_backbone
from .wsi_processor_fast import extract_wsi_features as extract_wsi_features_fast


def genereate_wsi_features(path=None, overwrite=False):
    # 0. 全局加载模型 (只加载一次)
    print("正在初始化 Virchow2 模型...")
    try:
        model, transform = get_virchow2_backbone(Config.VIRCHOW2_MODEL_ID, Config.DEVICE)
        print("模型加载完成。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    if path is not None:
        raw_data_root = path
    else:
        raw_data_root = Config.RAW_DATA_ROOT
    # 1. 遍历每个类别文件夹
    if not os.path.exists(raw_data_root):
        print(f"数据根目录不存在: {raw_data_root}")
        return

    class_dirs = [d for d in os.listdir(raw_data_root) if os.path.isdir(os.path.join(raw_data_root, d))]

    for class_name in class_dirs:
        if class_name not in Config.CLASS_MAP:
            print(f"Skipping unknown folder: {class_name}")
            continue

        class_path = os.path.join(raw_data_root, class_name)
        sample_names = os.listdir(class_path)

        for sample_name in tqdm(sample_names, desc=f"Extracting features for {class_name}"):
            sample_dir = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_dir): continue

            # --- 处理 SVS 文件 (processed 目录下) ---
            svs_dir = os.path.join(sample_dir, "processed")
            
            if os.path.exists(svs_dir):
                svs_files = glob.glob(os.path.join(svs_dir, "*.svs"))

                for svs_file in svs_files:
                    svs_filename = os.path.basename(svs_file)
                    svs_dir_path = os.path.dirname(svs_file)
                    save_name = os.path.splitext(svs_filename)[0] + ".pt"
                    save_path = os.path.join(svs_dir_path, save_name)

                    # 检查是否已存在
                    if os.path.exists(save_path):
                         # 如果不强制覆盖 且 (Config不强制覆盖)，则跳过
                         if not overwrite and not Config.OVERWRITE_SWI_FEATURES:
                             continue

                    # 传入预加载的模型和transform
                    if Config.USE_FAST_VERSION:
                        extract_wsi_features_fast(svs_file, save_path, model, transform)
                    else:
                        extract_wsi_features(svs_file, save_path, model, transform)


def generate_index_file(extract_features=False, overwrite=False, path=None, index_path=None):
    if extract_features:
        genereate_wsi_features(overwrite=overwrite, path=path)

    data_records = []
    
    if not os.path.exists(Config.RAW_DATA_ROOT):
        print(f"数据根目录不存在: {Config.RAW_DATA_ROOT}")
        return

    class_dirs = [d for d in os.listdir(Config.RAW_DATA_ROOT) if os.path.isdir(os.path.join(Config.RAW_DATA_ROOT, d))]
    print(f"Found classes: {class_dirs}")

    for class_name in class_dirs:
        if class_name not in Config.CLASS_MAP:
            print(f"Skipping unknown folder: {class_name}")
            continue

        label = Config.CLASS_MAP[class_name]
        class_path = os.path.join(Config.RAW_DATA_ROOT, class_name)
        sample_names = os.listdir(class_path)

        for sample_name in tqdm(sample_names, desc=f"Indexing {class_name}"):
            sample_dir = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_dir): continue

            # --- A. 查找 TXT 文件 ---
            txt_files = glob.glob(os.path.join(sample_dir, "*.txt"))
            txt_path = txt_files[1] if txt_files else ""

            # --- B. 查找 JPG 图片 ---
            jpg_files = glob.glob(os.path.join(sample_dir, "*.JPG"))
            jpg_paths_str = ";".join(jpg_files)

            # --- C. 查找已生成的 PT 文件 ---
            svs_dir = os.path.join(sample_dir, "processed")
            wsi_feat_paths = []

            if os.path.exists(svs_dir):
                # 查找 .pt 文件
                pt_files = glob.glob(os.path.join(svs_dir, "*.pt"))
                wsi_feat_paths.extend(pt_files)

            wsi_paths_str = ";".join(wsi_feat_paths)

            # --- D. 记录到列表 ---
            if txt_path or jpg_paths_str or wsi_paths_str:
                data_records.append({
                    "sample_id": sample_name,
                    "txt_path": txt_path,
                    "img_paths": jpg_paths_str,
                    "wsi_paths": wsi_paths_str,
                    "label": label
                })

    # 3. 生成 CSV
    index_path = index_path if index_path else Config.DATA_INDEX_PATH
    if data_records:
        df = pd.DataFrame(data_records)
        df.to_csv(index_path, index=False)
        print(f"Done! Index saved to {index_path}. Total samples: {len(df)}")
    else:
        print("No records found.")


if __name__ == "__main__":
    generate_index_file(extract_features=False)
