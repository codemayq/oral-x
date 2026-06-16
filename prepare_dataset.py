from mil.offline_feature_prepare import generate_index_file  as gif
from mil.offline_feature_prepare import genereate_wsi_features as gwf
from mil.utils import create_symlink_split, extract_txt_from_doc
from mil.utils import augment_minority_classes, clean_augmented_copies

import os
import pathlib


def doc2txt(data_dir: pathlib.Path):
    extract_txt_from_doc(data_dir)


def prepare_dataset(source_dir, target_dir, train_ratio=0.5, seed=42):
    create_symlink_split(source_dir, target_dir, train_ratio=train_ratio, seed=seed)


def augment_minority(target_dir, minority_classes, target_count, seed=42, clean_first=True):
    """
    扩充少数类样本到 target_count。
    通过对已有样本随机选取部分 JPG 图片创建软连接副本。

    Args:
        target_dir: 数据集根目录 (包含 train/ 子目录)
        minority_classes: 需要扩充的类别名列表
        target_count: 每个类别扩充到的目标样本数
        seed: 随机种子
        clean_first: 是否先清除之前的扩充副本
    """
    train_root = os.path.join(target_dir, "train")
    if clean_first:
        clean_augmented_copies(train_root, minority_classes)
    augment_minority_classes(train_root, minority_classes, target_count, seed=seed)


def generate_wsi_features(path=None, overwrite=False):
    gwf(path=path, overwrite=overwrite)


def generate_index_file(extract_features=False, overwrite=False, index_path=None):
    gif(extract_features=extract_features, overwrite=overwrite, index_path=index_path)


if __name__ == "__main__":
    doc2txt("/media/codingma/LLM/data-20260220/external-val")
    # generate_wsi_features(path="/media/codingma/LLM/data-20260220/internal-val", overwrite=False)
    # source_dir = "/media/codingma/LLM/data-20260220/train"
    # target_dir = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets-8csl-95_5"

    # Step 1: 划分 train/test
    # create_symlink_split(source_dir, target_dir, train_ratio=0.95, seed=42)

    # Step 2: 扩充 OSF 和 OSF+OLK 到与 OLK(56) 一致
    # augment_minority(
    #     target_dir=target_dir,
    #     minority_classes=["OSF", "OSF+OLK"],
    #     target_count=56,
    #     seed=42,
    # )

    # Step 3: 重新生成 index.csv (包含扩充后的样本)
    # generate_index_file(index_path=os.path.join(target_dir, "index.csv"))
