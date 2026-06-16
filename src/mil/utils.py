import os
import glob
import shutil
import random
import math
from tqdm import tqdm

import os
import random
import shutil

from .config import Config
from .doc2txt import doc2txt 


def extract_txt_from_doc(data_dir):
    doc2txt(data_dir)
    print(f"提取txt文件完成，文件位于doc源文件同位置下，同名txt文件")




def create_symlink_split(source_root, target_root, train_ratio=0.5, seed=42):
    """
    Creates symlinks for a train/test split from source_root to target_root.
    target_root should contain 'train' and 'test' directories.
    """
    random.seed(seed)
    
    # 确保目标目录存在 train 和 test
    train_root = os.path.join(target_root, "train")
    test_root = os.path.join(target_root, "test")
    
    if not os.path.exists(train_root):
        print(f"创建目录: {train_root}")
        os.makedirs(train_root)
    if not os.path.exists(test_root):
        print(f"创建目录: {test_root}")
        os.makedirs(test_root)

    # 获取源目录下的所有主类别文件夹
    categories = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    
    print(f"找到以下类别: {categories}")
    
    total_train = 0
    total_test = 0

    for category in categories:
        src_category_path = os.path.join(source_root, category)
        
        # 获取该类别下的样本文件夹
        samples = [s for s in os.listdir(src_category_path) if os.path.isdir(os.path.join(src_category_path, s))]
        
        # 随机打乱
        random.shuffle(samples)
        
        # 计算分割点
        current_train_ratio = train_ratio
        # if len(samples) < 50:
        #     print(f"类别 {category} 样本数 ({len(samples)}) 少于 40，强制设置 train_ratio 为 0.8")
        #     current_train_ratio = 0.8

        split_idx = int(len(samples) * current_train_ratio)
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        print(f"\n处理类别: {category}")
        print(f"  总样本数: {len(samples)}")
        print(f"  训练集数量: {len(train_samples)}")
        print(f"  测试集数量: {len(test_samples)}")
        
        # 创建软链接的辅助函数
        def make_links(sample_list, split_root):
            count = 0
            dst_category_path = os.path.join(split_root, category)
            os.makedirs(dst_category_path, exist_ok=True)
            
            for sample in sample_list:
                src_path = os.path.join(src_category_path, sample)
                dst_path = os.path.join(dst_category_path, sample)
                
                # 如果目标已存在（可能是之前的链接），先删除
                if os.path.exists(dst_path) or os.path.islink(dst_path):
                    try:
                        os.unlink(dst_path)
                    except IsADirectoryError:
                        # 如果确实是一个目录而不是软链接（不应该发生，但为了安全）
                        print(f"⚠️ 警告: 目标路径是一个实际目录，跳过: {dst_path}")
                        continue
                
                try:
                    os.symlink(src_path, dst_path)
                    count += 1
                except Exception as e:
                    print(f"❌ 创建软链接失败 {sample}: {e}")
            return count

        # 创建训练集链接
        t_count = make_links(train_samples, train_root)
        total_train += t_count
        
        # 创建测试集链接
        v_count = make_links(test_samples, test_root)
        total_test += v_count

    print(f"\n✅ 数据集划分完成！")
    print(f"总训练集样本链接数: {total_train}")
    print(f"总测试集样本链接数: {total_test}")




def augment_minority_classes(
    train_root: str,
    minority_classes: list,
    target_count: int,
    seed: int = 42,
):
    """
    通过软连接 + 随机选取部分 JPG 图片来扩充少数类样本。
    
    对于 minority_classes 中的每个类别，如果其样本数 < target_count，
    则从现有样本中随机选择并创建"副本"样本文件夹，
    每个副本随机选取原始样本中的一部分 JPG 图片进行软连接，
    其余文件（txt, docx, pt, kfb 等）全部软连接。
    
    Args:
        train_root:  训练集根目录，如 .../datasets-8csl-70_30/train
        minority_classes: 需要扩充的类别名列表，如 ["OSF", "OSF+OLK"]
        target_count: 扩充到的目标样本数
        seed: 随机种子
    """
    random.seed(seed)

    for cls_name in minority_classes:
        cls_path = os.path.join(train_root, cls_name)
        if not os.path.isdir(cls_path):
            print(f"[augment] 跳过不存在的类别目录: {cls_path}")
            continue

        existing_samples = [
            s for s in os.listdir(cls_path)
            if os.path.isdir(os.path.join(cls_path, s)) or os.path.islink(os.path.join(cls_path, s))
        ]
        # 过滤掉之前生成的副本（名称中含 -copy）
        original_samples = [s for s in existing_samples if "-copy" not in s]
        current_count = len(existing_samples)

        if current_count >= target_count:
            print(f"[augment] {cls_name}: 已有 {current_count} 个样本 >= 目标 {target_count}, 跳过")
            continue

        need = target_count - current_count
        print(f"[augment] {cls_name}: 现有 {current_count} (原始 {len(original_samples)}), 需扩充 {need} 个到 {target_count}")

        created = 0
        copy_idx = 1

        while created < need:
            # 从原始样本中随机选一个作为模板
            src_sample_name = random.choice(original_samples)
            src_sample_path = os.path.join(cls_path, src_sample_name)
            # 解析真实路径（因为样本本身可能就是软连接）
            real_src = os.path.realpath(src_sample_path)

            # 收集源样本内的所有文件
            all_files = os.listdir(real_src)
            jpg_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
            non_jpg_files = [f for f in all_files if not f.lower().endswith(('.jpg', '.jpeg'))]

            if len(jpg_files) < 2:
                # 图片太少就全部保留
                selected_jpgs = jpg_files[:]
            else:
                # 随机选取 [2, len-1] 张图片，保证有变化但不为空
                k = random.randint(2, max(2, len(jpg_files) - 1))
                selected_jpgs = random.sample(jpg_files, k)

            # 创建副本目录名: 原名-copyN
            copy_name = f"{src_sample_name}-copy{copy_idx}"
            copy_path = os.path.join(cls_path, copy_name)
            copy_idx += 1

            if os.path.exists(copy_path) or os.path.islink(copy_path):
                continue

            os.makedirs(copy_path, exist_ok=True)

            # 对选中的 JPG 创建软连接
            for jpg in selected_jpgs:
                src_file = os.path.join(real_src, jpg)
                dst_file = os.path.join(copy_path, jpg)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)

            # 对非 JPG 文件（txt, docx, kfb, json, pt, processed 等）全部软连接
            for f in non_jpg_files:
                src_file = os.path.join(real_src, f)
                dst_file = os.path.join(copy_path, f)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)

            created += 1

        final_count = len([
            s for s in os.listdir(cls_path)
            if os.path.isdir(os.path.join(cls_path, s)) or os.path.islink(os.path.join(cls_path, s))
        ])
        print(f"[augment] {cls_name}: 扩充完成, 最终样本数 {final_count}")


def clean_augmented_copies(train_root: str, minority_classes: list):
    """删除之前通过 augment_minority_classes 创建的所有 -copy 副本目录。"""
    for cls_name in minority_classes:
        cls_path = os.path.join(train_root, cls_name)
        if not os.path.isdir(cls_path):
            continue
        removed = 0
        for s in os.listdir(cls_path):
            if "-copy" in s:
                sp = os.path.join(cls_path, s)
                if os.path.isdir(sp) and not os.path.islink(sp):
                    shutil.rmtree(sp)
                    removed += 1
                elif os.path.islink(sp):
                    os.unlink(sp)
                    removed += 1
        if removed > 0:
            print(f"[clean] {cls_name}: 删除了 {removed} 个副本")


def check_processed_files():
    """
    检查已处理的 .pt 文件大小 (原有功能)
    """
    if not os.path.exists(Config.RAW_DATA_ROOT):
        print(f"Warning: Config.RAW_DATA_ROOT {Config.RAW_DATA_ROOT} does not exist.")
        return

    class_dirs = [d for d in os.listdir(Config.RAW_DATA_ROOT) if os.path.isdir(os.path.join(Config.RAW_DATA_ROOT, d))]
    
    for class_name in class_dirs:
        class_path = os.path.join(Config.RAW_DATA_ROOT, class_name)

        # 2. 遍历该类别下的 N 个样本文件夹
        sample_names = os.listdir(class_path)

        for sample_name in tqdm(sample_names, desc=f"Checking {class_name}"):
            sample_dir = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_dir): continue

            # --- C. 处理 SVS 文件 (processed 目录下) ---
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
                        file_size = os.path.getsize(save_path)
                        if file_size < 1024:
                            size_str = f"{file_size} B"
                        elif file_size < 1024**2:
                            size_str = f"{file_size / 1024:.2f} KB"
                        else:
                            size_str = f"{file_size / (1024**2):.2f} MB"
                        # print(f" existing: {save_path} (size: {size_str})")


if __name__ == "__main__":
    source_dir = "/media/codingma/LLM/data-1005"
    target_dir = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets"
    
    create_symlink_split(source_dir, target_dir)

