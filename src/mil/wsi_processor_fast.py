import openslide
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import Dataset, DataLoader

from .config import Config

# --- 1. 高效坐标生成 (新逻辑) ---
def get_tissue_coords_via_thumbnail(slide, patch_size, level=0, bg_threshold=240, step_size=None):
    """
    通过读取低倍率图像快速生成 Level 0 的组织坐标。
    代替了原先缓慢的串行扫描。
    """
    # 如果没指定步长，默认不重叠
    if step_size is None:
        step_size = patch_size

    # 1. 寻找合适的低倍率层级 (Downsample > 16x 比较合适，速度快且精度够)
    # 通常 Level 2 或 3 是 16x 或 32x 下采样
    seg_level = -1
    for idx, d in enumerate(slide.level_downsamples):
        if d >= 16: # 选取一个下采样率大于16的层级
            seg_level = idx
            break
    
    # 如果没有高倍下采样层，就强制用最后一层
    if seg_level == -1:
        seg_level = slide.level_count - 1

    # 获取该层级的尺寸和下采样倍率
    w_seg, h_seg = slide.level_dimensions[seg_level]
    downsample = slide.level_downsamples[seg_level]
    
    print(f"   Using segmentation level {seg_level} (Downsample: {downsample:.2f}x, Size: {w_seg}x{h_seg})")

    # 2. 读取整个低倍率图像 (IO 开销极小)
    # read_region 参数: location=(0,0) in level 0 ref, level=seg_level, size=(w, h)
    bg_image = slide.read_region((0, 0), seg_level, (w_seg, h_seg)).convert('RGB')
    bg_image_np = np.array(bg_image)

    # 3. 向量化计算 Mask (Numpy 操作，毫秒级)
    # 转换为灰度
    if bg_image_np.ndim == 3:
        gray = bg_image_np.mean(axis=2)
    else:
        gray = bg_image_np
        
    # 生成二值 Mask (True 为组织, False 为背景)
    # 可以在这里加入更复杂的形态学操作 (cv2.dilate/erode) 来去除孔洞
    tissue_mask = gray < bg_threshold

    # 4. 映射坐标
    # 我们需要在 seg_level 上遍历 patch，然后映射回 level 0
    # Level 0 上的 patch_size 在 seg_level 上对应的大小
    patch_size_seg = int(patch_size / downsample)
    step_size_seg = int(step_size / downsample)
    
    # 避免除以0或步长过小
    if patch_size_seg < 1: patch_size_seg = 1
    if step_size_seg < 1: step_size_seg = 1

    valid_coords = []
    
    # 在低倍率图上遍历
    # 这里的循环次数比原代码少几百倍
    rows = int(h_seg // step_size_seg)
    cols = int(w_seg // step_size_seg)

    for y in range(rows):
        for x in range(cols):
            # 获取当前小格子的坐标
            x_seg = x * step_size_seg
            y_seg = y * step_size_seg
            
            # 边界检查
            if x_seg + patch_size_seg > w_seg or y_seg + patch_size_seg > h_seg:
                continue

            # 快速检查 Mask: 如果该区域内组织的比例超过阈值 (例如 10% 是组织)
            # 或者简单点：只要中心点是组织
            patch_mask = tissue_mask[y_seg:y_seg+patch_size_seg, x_seg:x_seg+patch_size_seg]
            
            # 这里定义：如果Patch里有超过 1% 的像素是组织，则保留
            if np.mean(patch_mask) > 0.01:
                # 映射回 Level 0 坐标
                # 注意：OpenSlide read_region 始终需要 Level 0 的 (x, y)
                x_l0 = int(x_seg * downsample)
                y_l0 = int(y_seg * downsample)
                valid_coords.append((x_l0, y_l0))

    return valid_coords

# --- 2. 模型加载 ---
def get_virchow2_backbone(model_id, device):
    print(f"正在加载 Virchow2 模型: {model_id}")
    # (保持原样，省略部分代码以节省篇幅...)
    model = timm.create_model(
        model_id, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
    )
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    for param in model.parameters(): param.requires_grad = False
    model.to(device)
    model.eval()
    return model, transforms

def get_uni_backbone(local_dir, device):
    timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin")), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.to(device)
    model.eval()
    return model, transform



# --- 3. Dataset (只读，不判断) ---
class WsiPatchDataset(Dataset):
    def __init__(self, svs_path, coords, level, patch_size, transform=None):
        self.svs_path = svs_path
        self.coords = coords
        self.level = level
        self.patch_size = patch_size
        self.transform = transform
        self.slide = None # Lazy init

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.svs_path)

        x, y = self.coords[idx]
        
        try:
            # 直接读取，不做背景判断，因为 coords 已经是过滤过的了
            img = self.slide.read_region(
                (x, y),
                self.level,
                (self.patch_size, self.patch_size)
            ).convert('RGB')

            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            # 异常处理：返回零张量，确保 Batch 不会崩
            # 更好的做法可能是记录 log
            print(f"Read Error: {e}")
            return torch.zeros((3, self.patch_size, self.patch_size))

# --- 4. 核心流程 ---
def extract_wsi_features(svs_path, save_path, model, transform, config=Config()):
    device = config.DEVICE
    slide_id = os.path.splitext(os.path.basename(svs_path))[0]
    
#    if os.path.exists(save_path):
#        print(f"[{slide_id}] 结果已存在，跳过。")
#        return True

    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception as e:
        print(f"无法打开 SVS: {e}")
        return False

    # --- 步骤 1: 极速坐标生成 ---
    print(f"[{slide_id}] 正在生成组织坐标 (Low-Res Masking)...")
    
    # 假设 config.PATCH_SIZE 是 Level 0 的尺寸 (例如 224 或 256)
    # 假设 config.TILE_LEVEL 是读取数据的层级 (通常是 0)
    # 如果我们要提取 Level 0 的特征，传入 level=0
    valid_coords = get_tissue_coords_via_thumbnail(
        slide, 
        patch_size=config.PATCH_SIZE, 
        level=config.TILE_LEVEL, 
        bg_threshold=config.BG_THRESHOLD
    )
    
    # 获取完坐标后暂时不需要主线程的 slide 了，但 dataset 里会重新开
    # 这里不 close slide，因为 get_tissue_coords 用的引用，但也无所谓
    
    if not valid_coords:
        print(f"[{slide_id}] 未发现组织区域，跳过。")
        slide.close()
        return False

    print(f"[{slide_id}] 发现 {len(valid_coords)} 个有效 Patch。准备推理...")

    # --- 步骤 2: DataLoader 并行读取 ---
    dataset = WsiPatchDataset(
        svs_path=svs_path,
        coords=valid_coords,
        level=config.TILE_LEVEL,
        patch_size=config.PATCH_SIZE,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE_WSI,
        shuffle=False,
        num_workers=config.NUM_WORKERS, 
        pin_memory=True,
        prefetch_factor=2 # 预取2个batch，保证 GPU 不空闲
    )

    all_tokens = []

    # --- 步骤 3: 混合精度推理 ---
    try:
        with torch.inference_mode():
            for img_batch in tqdm(dataloader, desc=f"Extracting {slide_id}"):
                img_batch = img_batch.to(device, non_blocking=True) # non_blocking 加速传输
                
                with torch.amp.autocast('cuda', enabled=config.USE_AMP, dtype=torch.float16):
                    output = model(img_batch)
                    
                    # Virchow2 特征处理逻辑
                    class_token = output[:, 0]
                    patch_tokens = output[:, 5:]
                    embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                    
                    # 立即转回 CPU 并转为半精度保存内存
                    all_tokens.append(embedding.detach().cpu().to(torch.float16))
                    
    except Exception as e:
        print(f"推理中断: {e}")
        slide.close()
        return False

    slide.close()

    # --- 步骤 4: 保存 ---
    if all_tokens:
        final_tokens = torch.cat(all_tokens, dim=0)
        
        # 也可以保存坐标信息，方便后续可视化
        if config.SAVE_COORDS:
            save_data = {
                "features": final_tokens,
                "coords": np.array(valid_coords)
            }
        else:
            save_data = final_tokens
        
        if save_path is not None:
            torch.save(save_data, save_path)
            print(f"[{slide_id}] 完成。Shape: {final_tokens.shape} -> {save_path}")
            return save_data
        else:
            return save_data
    
    raise RuntimeError("extract_wsi_features failed")


# =========================================================
# 5. Example Entry
# =========================================================
if __name__ == "__main__":
    config = Config()

    model, transform = get_virchow2_backbone(
        model_id=config.VIRCHOW2_MODEL_ID,
        device=config.DEVICE
    )

    svs_path = "/media/codingma/LLM/lcx/data-1005/已整理-OLK/傅雅婷OLK/processed/傅雅婷1-中.svs"
    save_path = "/media/codingma/LLM/lcx/data-1005/已整理-OLK/傅雅婷OLK/processed/傅雅婷1-中-fast.pt"

    extract_wsi_features(
        svs_path,
        save_path,
        model,
        transform,
        config
    )
