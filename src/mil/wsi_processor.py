import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import Dataset, DataLoader

from .config import Config


# --- 辅助函数 ---
def is_tissue(img_patch, bg_threshold):
    """判断切片是否为组织区域 (简单亮度过滤)"""
    np_img = np.array(img_patch)
    if np_img.size == 0 or np_img.ndim < 3: return False
    # 转换为灰度或取平均亮度
    avg_intensity = np.mean(np_img[:, :, :3])
    # 背景通常是白色的（值接近255），组织较暗
    return avg_intensity < bg_threshold


def get_virchow2_backbone(model_id, device):
    """实例化并加载 Virchow2 权重，并返回模型和 Transform"""
    print(f"正在加载 Virchow2 模型: {model_id}")

    model = timm.create_model(
        model_id,
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU
    )

    # 获取 Virchow2 要求的 Transform
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # 冻结权重
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)
    model.eval()

    return model, transforms


# --- 自定义 Dataset 实现并行读取 ---
class WsiPatchDataset(Dataset):
    def __init__(self, svs_path, coords, level, patch_size, transform=None):
        """
        args:
            svs_path: SVS 文件路径
            coords: 有效 patch 的坐标列表 [(x, y), ...]
            level: 读取的缩放层级
            patch_size: patch 大小
            transform: 预处理变换
        """
        self.svs_path = svs_path
        self.coords = coords
        self.level = level
        self.patch_size = patch_size
        self.transform = transform
        # 每个 worker 拥有独立的 OpenSlide 对象
        self.slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # 确保每个 worker 初始化自己的 slide 对象（OpenSlide 不支持跨进程共享）
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.svs_path)

        x, y = self.coords[idx]

        try:
            img = self.slide.read_region(
                (x, y),
                self.level,
                (self.patch_size, self.patch_size)
            ).convert('RGB')

            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error reading region ({x}, {y}): {e}")
            # 返回全黑图像防止崩溃，实际场景可能需要更好处理
            return torch.zeros((3, self.patch_size, self.patch_size))

    def close(self):
        if self.slide is not None:
            self.slide.close()


# --- 核心提取函数 ---
def extract_wsi_features(svs_path, save_path, model, transform, config=Config()):
    """
    执行 WSI 切片、过滤和特征提取流程
    Args:
        svs_path: WSI 路径
        save_path: 结果保存路径
        model: 预加载的模型 (避免重复加载)
        transform: 预加载的 transform
        config: 配置对象
    """
    device = config.DEVICE
    slide_id = os.path.splitext(os.path.basename(svs_path))[0]

    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception as e:
        print(f"无法打开 SVS 文件 {svs_path}: {e}")
        return False

    w, h = slide.level_dimensions[0]
    valid_coords = []

    # 1. 快速扫描有效区域 (Coordinate collecting)
    # 为了加快速度，这一步仍然是串行的，但只做轻量级读取或基于低倍率图做 mask
    # 这里保持原有逻辑，但只存坐标

    print(f"[{slide_id}] 正在扫描组织区域...")
    for x in range(0, w, config.PATCH_SIZE):
        for y in range(0, h, config.PATCH_SIZE):
            if x + config.PATCH_SIZE > w or y + config.PATCH_SIZE > h: continue

            # 读取小块进行背景判断
            # 优化：只读不做 transform，且可以考虑 read_region 耗时
            patch = slide.read_region((x, y), config.TILE_LEVEL, (config.PATCH_SIZE, config.PATCH_SIZE))
            if is_tissue(patch, config.BG_THRESHOLD):
                valid_coords.append((x, y))

    slide.close()  # 主线程关闭 slide

    if not valid_coords:
        print(f"[{slide_id}] 未发现组织，跳过。")
        return False

    print(f"[{slide_id}] 发现 {len(valid_coords)} 个有效 Patch，开始特征提取...")

    # 2. 构建 Dataset 和 DataLoader
    dataset = WsiPatchDataset(
        svs_path=svs_path,
        coords=valid_coords,
        level=config.TILE_LEVEL,
        patch_size=config.PATCH_SIZE,
        transform=transform
    )

    # num_workers > 0 开启多进程并行读取，pin_memory=True 加速 CPU->GPU 传输
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE_WSI,
        shuffle=False,
        num_workers=config.NUM_WORKERS,  # 使用配置中的 workers 数量
        pin_memory=True
    )

    all_tokens = []

    # 3. 批量推理
    try:
        # 使用 inference_mode 代替 no_grad，性能更好
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Extracting {slide_id}"):
                img_batch = batch.to(device)
                
                # 使用 Mixed Precision (AMP)
                with torch.amp.autocast(device, enabled=config.USE_AMP, dtype=torch.float16):
                    output = model(img_batch) # (B, 261, 1280)
                    
                    # 按照 Virchow2 官方文档推荐方式处理:
                    # Class Token (idx 0) + Mean Patch Tokens (idx 5:)
                    # 索引 1-4 是 Register tokens，跳过
                    class_token = output[:, 0]      # (B, 1280)
                    patch_tokens = output[:, 5:]    # (B, 256, 1280)
                    
                    # 拼接: (B, 2560)
                    embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                    
                    # 转为 FP16 节省存储空间 (1280*2 * 2bytes = 5KB/patch)
                    embedding = embedding.to(torch.bfloat16)
                
                all_tokens.append(embedding.cpu())
    except Exception as e:
        print(f"特征提取中断: {e}")
        return False

    # 4. 保存结果
    if all_tokens:
        final_tokens = torch.cat(all_tokens, dim=0)  # (Total_Patches, 261, 1280)
        print(save_path, "->", final_tokens.shape)
        torch.save(final_tokens, save_path)
        print(f"[{slide_id}] 保存特征: {final_tokens.shape} -> {save_path}")
        return True
    else:
        return False