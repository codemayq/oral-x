import os
import sys
import csv
import glob
import torch
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report, multilabel_confusion_matrix,
    confusion_matrix,
)

from mil.config import Config, map_label_for_3cls_training
from mil.inference import InferencePipeline
from mil.inference_moe import MoEInferencePipeline

# ==================== 消融实验：7 种模态组合 ====================
MODALITY_COMBINATIONS = [
    {"use_txt": True,  "use_img": False, "use_svs": False, "tag": "txt"},
    {"use_txt": False, "use_img": True,  "use_svs": False, "tag": "img"},
    {"use_txt": False, "use_img": False, "use_svs": True,  "tag": "svs"},
    {"use_txt": True,  "use_img": True,  "use_svs": False, "tag": "txt_img"},
    {"use_txt": True,  "use_img": False, "use_svs": True,  "tag": "txt_svs"},
    {"use_txt": False, "use_img": True,  "use_svs": True,  "tag": "img_svs"},
    {"use_txt": True,  "use_img": True,  "use_svs": True,  "tag": "txt_img_svs"},
]


# ==================== Config 切换 ====================

def _configure_for_3cls():
    """将 Config 切换为 3 分类单标签模式"""
    Config.NUM_CLASSES = 3
    Config.MULTI_LABEL = False
    Config.TARGET_CLASS_NAMES = Config.MAJOR_CLASS_NAMES
    Config.THREE_CLS_COARSE_FOLDERS = False
    Config.MAP_RAW_LABEL_TO_MAJOR_3CLS = True
    # 测试集使用 8 类文件夹 + Config.CLASS_MAP（细类 id），与 mil.config / mil.dataset 一致
    Config.FUSION_DIM = 1280
    Config.FUSION_LAYERS = 12
    Config.FUSION_HEADS = 16
    Config.FUSION_DROPOUT = 0.1
    Config.CHECKPOINT_DIR = (
        # "/media/codingma/LLM/lcx/Medical_Info_Classification/get_log_checkpoints_70_30_single_label_3cls_202620406"
        # "checkpoints_70_30_single_label_3cls_20262027"
        "/media/codingma/LLM/lcx/Medical_Info_Classification/revisied_ckpt_20260602_3cls_v3"
    )


def _configure_for_7cls():
    """将 Config 切换为 7 分类多标签模式"""
    Config.NUM_CLASSES = 7
    Config.MULTI_LABEL = True
    Config.TARGET_CLASS_NAMES = Config.MULTI_LABEL_CLASS_NAMES
    Config.CLASS_MAP = Config.RAW_CLASS_MAP
    Config.FUSION_DIM = 1280
    Config.FUSION_LAYERS = 12
    Config.FUSION_HEADS = 16
    Config.FUSION_DROPOUT = 0.1
    Config.CHECKPOINT_DIR = (
        # "/media/codingma/LLM/lcx/Medical_Info_Classification/get_log_checkpoints_70_30_multi_label_7cls_202620406"
        # "checkpoints_70_30_multi_label_8cls_202620302"
        # "/media/codingma/LLM/lcx/Medical_Info_Classification/revisied_ckpt_20260527_8cls"
        "/media/codingma/LLM/lcx/Medical_Info_Classification/revisied_ckpt_20260601_8cls_v3"
    )


# ==================== 工具函数 ====================

def _build_pipeline(checkpoint_dir):
    """根据当前 Config 状态构建推理管线（自动检测 MoE）"""
    checkpoint_path = os.path.join(checkpoint_dir, "best_val.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_dir, "best_val_moe.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    temp_ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in temp_ckpt:
        state_keys = temp_ckpt["state_dict"].keys()
    elif "model_state_dict" in temp_ckpt:
        state_keys = temp_ckpt["model_state_dict"].keys()
    else:
        state_keys = temp_ckpt.keys()

    is_moe = any("moe_classifier" in k or "moe_ffn" in k for k in state_keys)

    if is_moe:
        print("Detected MoE model checkpoint.")
        pipeline = MoEInferencePipeline(checkpoint_path)
    else:
        print("Detected Standard model checkpoint.")
        pipeline = InferencePipeline(checkpoint_path)

    return pipeline


def _save_predictions_csv(csv_path, results_log, class_names):
    """将每个样本的预测结果、标签、各类别概率保存到 CSV。"""
    prob_headers = [f"prob_{name}" for name in class_names]
    header = ["sample_name", "true_label", "pred_label", "correct"] + prob_headers

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results_log:
            line = [
                row["sample"],
                row["true_str"],
                row["pred_str"],
                int(row["correct"]),
            ] + [f"{p:.6f}" for p in row["probs"]]
            writer.writerow(line)
    print(f"  -> CSV saved: {csv_path}")


def _collect_samples(test_root, use_txt, use_img, use_svs):
    """根据模态开关收集测试样本"""
    samples = []

    multi_label_mapping = {
        0: [0],  1: [1],  2: [2],  3: [3],
        4: [0, 3],  5: [4],  6: [5],  7: [6],
    }

    raw_class_map = Config.CLASS_MAP

    for class_name in sorted(os.listdir(test_root)):
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in raw_class_map:
            print(f"  Skipping unknown class folder: {class_name}")
            continue

        raw_label_id = raw_class_map[class_name]

        if Config.MULTI_LABEL:
            true_label_indices = multi_label_mapping.get(raw_label_id, [])
            label_vec = np.zeros(Config.NUM_CLASSES, dtype=int)
            label_vec[true_label_indices] = 1
            label_storage = label_vec
        else:
            label_storage = map_label_for_3cls_training(raw_label_id)

        for sample_name in sorted(os.listdir(class_dir)):
            sample_dir = os.path.join(class_dir, sample_name)
            if not os.path.isdir(sample_dir):
                continue

            text_content = ""
            if use_txt:
                txt_files = glob.glob(os.path.join(sample_dir, "*.txt"))
                # print(txt_files)
                # breakpoint()
                if txt_files:
                    with open(txt_files[0], "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()

            img_paths = []
            if use_img:
                for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg"]:
                    img_paths.extend(glob.glob(os.path.join(sample_dir, ext)))

            svs_paths = []
            if use_svs and os.path.exists(os.path.join(sample_dir, "processed")):
                svs_paths = glob.glob(os.path.join(sample_dir, "processed", "*.pt"))

            samples.append({
                "path": sample_dir,
                "class_name": class_name,
                "label": label_storage,
                "text": text_content,
                "images": img_paths,
                "svs": svs_paths,
            })

    return samples


# ==================== 核心推理函数 ====================

def evaluate_single_combination(pipeline, test_root, use_txt, use_img, use_svs,
                                results_save_dir, prob_threshold=0.7,
                                enable_fallback=True):
    """对单个模态组合执行推理，保存 CSV，打印指标到终端。"""

    class_names = Config.TARGET_CLASS_NAMES
    modality_tag = []
    if use_txt:
        modality_tag.append("MRT")
    if use_img:
        modality_tag.append("Clinical_images")
    if use_svs:
        modality_tag.append("WSI")
    tag_str = "_".join(modality_tag)

    print(f"\n{'='*70}")
    print(f"  Modality : {tag_str}")
    print(f"  Classes  : {Config.NUM_CLASSES}  "
          f"({'Multi-Label' if Config.MULTI_LABEL else 'Single-Label'})")
    print(f"  Test root: {test_root}")
    print(f"{'='*70}")

    samples = _collect_samples(test_root, use_txt, use_img, use_svs)
    print(f"  Found {len(samples)} test samples.")
    if len(samples) == 0:
        return

    y_true, y_pred, y_probs = [], [], []
    results_log = []

    for sample in tqdm(samples, desc=f"[{tag_str}]"):
        try:
            probs = pipeline.predict_proba(
                sample["text"], sample["images"], sample["svs"]
            )

            if Config.MULTI_LABEL:
                pred_vec = (probs > prob_threshold).astype(int)
                is_fallback = False
                if enable_fallback and pred_vec.sum() == 0:
                    pred_vec[np.argmax(probs)] = 1
                    is_fallback = True

                y_true.append(sample["label"])
                y_pred.append(pred_vec)
                y_probs.append(probs)

                pred_names = [class_names[i] for i, v in enumerate(pred_vec) if v == 1]
                true_names = [class_names[i] for i, v in enumerate(sample["label"]) if v == 1]
                is_correct = np.array_equal(sample["label"], pred_vec)
                true_str = ";".join(true_names)
                pred_str = ";".join(pred_names)
            else:
                pred_idx = int(np.argmax(probs))
                true_idx = int(sample["label"])

                y_true.append(true_idx)
                y_pred.append(pred_idx)
                y_probs.append(probs)

                true_str = class_names[true_idx]
                pred_str = class_names[pred_idx]
                is_correct = (true_idx == pred_idx)
                is_fallback = False

            status = "OK" if is_correct else "WRONG"
            if is_fallback:
                status += " (Fallback)"

            tqdm.write(f"  [{status}] {os.path.basename(sample['path'])}"
                       f"  True: {true_str}  Pred: {pred_str}")

            results_log.append({
                "sample": os.path.basename(sample["path"]),
                "true_str": true_str,
                "pred_str": pred_str,
                "correct": is_correct,
                "probs": probs,
            })

        except Exception as e:
            print(f"  Error on {sample['path']}: {e}")

    # ---- 保存 CSV ----
    csv_filename = f"{tag_str}_{time.strftime('%Y%m%d_%H%M%S')}_predictions.csv"
    csv_path = os.path.join(results_save_dir, csv_filename)
    _save_predictions_csv(csv_path, results_log, class_names)

    # ---- 打印指标 ----
    y_probs_arr = np.array(y_probs)

    if Config.MULTI_LABEL:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        acc = accuracy_score(y_true_arr, y_pred_arr)
        correct_count = int(np.sum(np.all(y_true_arr == y_pred_arr, axis=1)))
        total_count = len(y_true_arr)

        print(f"\n  Exact Match Accuracy: {acc:.4f} ({correct_count}/{total_count})")
        print("\n  Classification Report:")
        try:
            print(classification_report(
                y_true_arr, y_pred_arr,
                target_names=class_names, digits=4, zero_division=0,
            ))
        except Exception as e:
            print(f"  Error generating report: {e}")

        print("  Multilabel Confusion Matrix:")
        try:
            mcm = multilabel_confusion_matrix(y_true_arr, y_pred_arr)
            for ci, cn in enumerate(class_names):
                print(f"    Class: {cn}")
                print(f"    {mcm[ci]}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        acc = accuracy_score(y_true_arr, y_pred_arr)
        correct_count = int((y_true_arr == y_pred_arr).sum())
        total_count = len(y_true_arr)

        print(f"\n  Accuracy: {acc:.4f} ({correct_count}/{total_count})")
        print("\n  Classification Report:")
        try:
            print(classification_report(
                y_true_arr, y_pred_arr,
                target_names=class_names, digits=4, zero_division=0,
            ))
        except Exception as e:
            print(f"  Error generating report: {e}")

        print("  Confusion Matrix:")
        try:
            cm = confusion_matrix(y_true_arr, y_pred_arr)
            print(f"    {cm}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"{'='*70}\n")


# ==================== 主入口 ====================

def main_3cls(test_root, results_dir):
    """3 分类单标签 — 遍历 7 种模态组合进行消融实验"""
    _configure_for_3cls()

    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("#  Ablation Study — 3-Class Single-Label")
    print("#  Test dataset: " + test_root)
    print("#  Results dir : " + results_dir)
    print("#" * 70)

    pipeline = _build_pipeline(Config.CHECKPOINT_DIR)

    for combo in MODALITY_COMBINATIONS:
        evaluate_single_combination(
            pipeline, test_root,
            use_txt=combo["use_txt"],
            use_img=combo["use_img"],
            use_svs=combo["use_svs"],
            results_save_dir=results_dir,
        )

    print("\n>>> 3-class ablation finished. All CSVs saved to:", results_dir)


def main_7cls(test_root, results_dir):
    """7 分类多标签 — 遍历 7 种模态组合进行消融实验"""
    _configure_for_7cls()
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("#  Ablation Study — 7-Class Multi-Label")
    print("#  Test dataset: " + test_root)
    print("#  Results dir : " + results_dir)
    print("#" * 70)

    pipeline = _build_pipeline(Config.CHECKPOINT_DIR)

    for combo in MODALITY_COMBINATIONS:
        evaluate_single_combination(
            pipeline, test_root,
            use_txt=combo["use_txt"],
            use_img=combo["use_img"],
            use_svs=combo["use_svs"],
            results_save_dir=results_dir,
        )

    print("\n>>> 7-class ablation finished. All CSVs saved to:", results_dir)


class ResNetGradCAM:
    """
    Grad-CAM visualizer for the ResNet101 image branch.
    Hooks into pipeline.model.img_backbone to capture activations and gradients,
    then generates class-specific heatmaps overlaid on original images.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = pipeline.device
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    # ---- hooks ----
    def _fwd_hook(self, module, inp, out):
        self._activations = out  # (N, 2048, h, w) — 保留计算图以便 backward

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]  # (N, 2048, h, w)

    # ---- public API ----
    def __call__(
        self,
        text_input: str,
        image_paths: list,
        svs_paths: list,
        save_dir: Path,
        sample_id: str,
        target_class: Optional[int] = None,
    ) -> None:
        """
        对单个样本生成 Grad-CAM 热力图并保存。
        """
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            return

        # 注册 forward / backward hook
        fwd_h = self.model.img_backbone.register_forward_hook(self._fwd_hook)
        bwd_h = self.model.img_backbone.register_full_backward_hook(self._bwd_hook)

        try:
            # ---- 1. 准备输入（与 InferencePipeline.predict_proba 保持一致）----
            text_enc = self.pipeline.tokenizer(
                text_input if text_input else "",
                max_length=Config.MAX_TEXT_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            input_ids = text_enc['input_ids'].to(self.device)
            attention_mask = text_enc['attention_mask'].to(self.device)

            img_tensors: list = []
            loaded_paths: list = []
            for p in valid_paths:
                try:
                    img = PILImage.open(p).convert('RGB')
                    img_tensors.append(self.pipeline.normal_transform(img))
                    loaded_paths.append(p)
                except Exception:
                    continue
            if not img_tensors:
                return

            normal_imgs = torch.stack(img_tensors).unsqueeze(0).to(self.device)  # (1, N, 3, 224, 224)

            # WSI features
            wsi_feat_list: list = []
            for sp in svs_paths:
                if not os.path.exists(sp) or not sp.endswith('.pt'):
                    continue
                try:
                    feat = torch.load(sp, map_location='cpu')
                    if feat.ndim == 3:
                        feat = feat.view(-1, feat.size(-1))
                    wsi_feat_list.append(feat)
                except Exception:
                    continue

            if wsi_feat_list:
                wsi_feat = torch.cat(wsi_feat_list, dim=0).unsqueeze(0).to(self.device)
                wsi_mask = torch.ones(1, wsi_feat.size(1)).to(self.device)
            else:
                wsi_feat = torch.zeros(1, 1, Config.WSI_INPUT_DIM).to(self.device)
                wsi_mask = torch.zeros(1, 1).to(self.device)

            # ---- 2. 带梯度的前向 + 反向 ----
            self.model.eval()
            with torch.enable_grad():
                with torch.amp.autocast(self.device.type, enabled=Config.USE_AMP, dtype=torch.float16):
                    logits = self.model(input_ids, attention_mask, normal_imgs, wsi_feat, wsi_mask)
                    probs = torch.sigmoid(logits)

                tc = target_class if target_class is not None else int(probs[0].argmax().item())
                self.model.zero_grad()
                logits[0, tc].backward()

            if self._activations is None or self._gradients is None:
                print(f"[GradCAM] 未捕获到特征图/梯度 (sample={sample_id})，跳过。")
                return

            # ---- 3. 计算 Grad-CAM ----
            acts = self._activations.detach().float()   # (N, 2048, h, w)
            grads = self._gradients.detach().float()    # (N, 2048, h, w)

            weights = grads.mean(dim=[2, 3], keepdim=True)            # (N, 2048, 1, 1)
            cam = torch.relu((weights * acts).sum(dim=1)).cpu().numpy()  # (N, h, w)

            # ---- 4. 生成并保存热力图 ----
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            pred_name = Config.TARGET_CLASS_NAMES[tc]

            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Serif CJK JP', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception:
                pass

            for i, img_path in enumerate(loaded_paths):
                orig = cv2.imread(img_path)
                if orig is None:
                    continue
                orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                h, w = orig_rgb.shape[:2]

                # 归一化 CAM 到 [0, 1]
                c = cam[i]
                c = c - c.min()
                if c.max() > 0:
                    c = c / c.max()
                cam_resized = cv2.resize(c, (w, h))

                # 伪彩色 + 叠加
                heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                overlay = np.uint8(0.6 * orig_rgb + 0.4 * heatmap_rgb)

                # 绘制三联图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(orig_rgb)
                axes[0].set_title("Original", fontsize=14)
                axes[0].axis('off')

                axes[1].imshow(cam_resized, cmap='jet')
                axes[1].set_title(f"Grad-CAM (class: {pred_name})", fontsize=14)
                axes[1].axis('off')

                axes[2].imshow(overlay)
                axes[2].set_title("Overlay", fontsize=14)
                axes[2].axis('off')

                img_stem = Path(img_path).stem
                # plt.suptitle(f"Sample: {sample_id}  |  Image: {img_stem}", fontsize=16)
                plt.tight_layout()
                
                # 按照用户要求，直接使用原始图片的相对路径作为保存路径
                # 这里的 save_dir 已经是目标文件夹路径了，我们只需要保存为 {img_stem}.png
                fname = f"{img_stem}.png"
                plt.savefig(str(save_dir / fname), dpi=150, bbox_inches='tight')
                plt.close(fig)

                # 保留拼接图；同级以 stem 为名的文件夹内存三张原始分辨率大图（避免 subplot 导出缩放）
                sub_dir = save_dir / img_stem
                sub_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sub_dir / "original.png"), cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sub_dir / "heatmap.png"), heatmap_bgr)
                cv2.imwrite(str(sub_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        finally:
            fwd_h.remove()
            bwd_h.remove()
            self._activations = None
            self._gradients = None
            self.model.zero_grad()


def main_heatmap(test_root="/media/codingma/LLM/lcx/Medical_Info_Classification/datasets/test",
                 results_dir="/media/codingma/LLM/results/figures-single-label/task8_heatmap"):
    """提取全模态全开的情况下，图片分支的特征并绘制类似的特征热图"""
    _configure_for_7cls() 
    
    print("\n" + "#" * 70)
    print("#  Task: Heatmap Extraction (ResNet Branch)")
    print("#  Test dataset: " + test_root)
    print("#  Results dir : " + results_dir)
    print("#" * 70)

    pipeline = _build_pipeline(Config.CHECKPOINT_DIR)
    gradcam = ResNetGradCAM(pipeline)

    # 全模态全开
    use_txt, use_img, use_svs = True, True, True
    samples = _collect_samples(test_root, use_txt, use_img, use_svs)
    print(f"  Found {len(samples)} test samples.")
    if len(samples) == 0:
        return

    # 确定 test_root 的父目录，用于计算相对路径
    # 例如 test_root 是 /media/codingma/LLM/data-20260220/internal-val
    # 我们想要保留 internal-val/Fibroma/陈理司/DSC_2365.png
    # 那么 base_dir 应该是 /media/codingma/LLM/data-20260220
    base_dir = os.path.dirname(test_root)

    for sample in tqdm(samples, desc="[Heatmap]"):
        try:
            # 获取样本的相对路径部分，例如 internal-val/Fibroma/陈理司
            sample_rel_dir = os.path.relpath(sample["path"], base_dir)
            save_dir = os.path.join(results_dir, sample_rel_dir)
            
            sample_id = os.path.basename(sample["path"])
            
            gradcam(
                text_input=sample["text"],
                image_paths=sample["images"],
                svs_paths=sample["svs"],
                save_dir=Path(save_dir),
                sample_id=sample_id,
            )
        except Exception as e:
            print(f"  Error on {sample['path']}: {e}")

    print("\n>>> Heatmap extraction finished. Saved to:", results_dir)


class WSIHeatmapVisualizer:
    """
    Attention visualizer for the WSI branch.
    Hooks into pipeline.model.wsi_mil.attention_weights to capture attention scores,
    then generates heatmaps overlaid on the WSI images.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = pipeline.device
        self._attention_weights: Optional[torch.Tensor] = None

    def _fwd_hook(self, module, inp, out):
        self._attention_weights = out.detach()  # (B, N, 1)

    def __call__(
        self,
        text_input: str,
        image_paths: list,
        svs_paths: list,
        save_dir: Path,
        sample_id: str,
        target_class: Optional[int] = None,
    ) -> None:
        if not svs_paths:
            return

        import openslide
        from mil.wsi_processor_fast import get_tissue_coords_via_thumbnail

        # 注册 forward hook
        hook_handle = self.model.wsi_mil.attention_weights.register_forward_hook(self._fwd_hook)

        try:
            # ---- 1. 准备输入 ----
            text_enc = self.pipeline.tokenizer(
                text_input if text_input else "",
                max_length=Config.MAX_TEXT_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            input_ids = text_enc['input_ids'].to(self.device)
            attention_mask = text_enc['attention_mask'].to(self.device)

            img_tensors: list = []
            for p in image_paths:
                if os.path.exists(p):
                    try:
                        img = PILImage.open(p).convert('RGB')
                        img_tensors.append(self.pipeline.normal_transform(img))
                    except Exception:
                        pass
            if img_tensors:
                normal_imgs = torch.stack(img_tensors).unsqueeze(0).to(self.device)
            else:
                normal_imgs = torch.zeros(1, 1, 3, 224, 224).to(self.device)

            # WSI features
            wsi_feat_list: list = []
            valid_svs_paths: list = []
            for sp in svs_paths:
                if not os.path.exists(sp) or not sp.endswith('.pt'):
                    continue
                try:
                    feat = torch.load(sp, map_location='cpu')
                    if feat.ndim == 3:
                        feat = feat.view(-1, feat.size(-1))
                    wsi_feat_list.append(feat)
                    valid_svs_paths.append(sp)
                except Exception:
                    continue

            if not wsi_feat_list:
                return

            wsi_feat = torch.cat(wsi_feat_list, dim=0).unsqueeze(0).to(self.device)
            wsi_mask = torch.ones(1, wsi_feat.size(1)).to(self.device)

            # ---- 2. 前向传播 ----
            self.model.eval()
            with torch.no_grad():
                with torch.amp.autocast(self.device.type, enabled=Config.USE_AMP, dtype=torch.float16):
                    logits = self.model(input_ids, attention_mask, normal_imgs, wsi_feat, wsi_mask)
                    probs = torch.sigmoid(logits)

            tc = target_class if target_class is not None else int(probs[0].argmax().item())
            pred_name = Config.TARGET_CLASS_NAMES[tc]

            if self._attention_weights is None:
                print(f"[WSIHeatmap] 未捕获到 Attention Weights (sample={sample_id})，跳过。")
                return

            # ---- 3. 计算 Attention 热图 ----
            # (1, N, 1) -> (N,)
            A_before_softmax = self._attention_weights[0, :, 0].float()
            # 这里的 softmax 是在所有切片的 token 上一起做的，和模型里的逻辑一致
            A = torch.softmax(A_before_softmax, dim=0).cpu().numpy()

            # ---- 4. 拆分每个切片并绘制 ----
            start_idx = 0
            for sp, feat in zip(valid_svs_paths, wsi_feat_list):
                num_tokens = feat.size(0)
                slice_A = A[start_idx : start_idx + num_tokens]
                start_idx += num_tokens

                svs_img_path = sp.replace('.pt', '.svs')
                if not os.path.exists(svs_img_path):
                    print(f"[WSIHeatmap] 找不到对应的 SVS 文件: {svs_img_path}")
                    continue

                try:
                    slide = openslide.OpenSlide(svs_img_path)
                    valid_coords = get_tissue_coords_via_thumbnail(
                        slide, 
                        patch_size=Config.PATCH_SIZE, 
                        level=Config.TILE_LEVEL, 
                        bg_threshold=Config.BG_THRESHOLD
                    )
                except Exception as e:
                    print(f"[WSIHeatmap] 处理 SVS 失败 {svs_img_path}: {e}")
                    continue

                if len(valid_coords) != num_tokens:
                    print(f"[WSIHeatmap] 坐标数量 ({len(valid_coords)}) 与特征数量 ({num_tokens}) 不匹配: {svs_img_path}")
                    continue

                # 寻找不超过 5000x5000 的最大分辨率 level
                target_level = 0
                for i, (w, h) in enumerate(slide.level_dimensions):
                    if w <= 5000 and h <= 5000:
                        target_level = i
                        break
                
                w_l, h_l = slide.level_dimensions[target_level]
                downsample = slide.level_downsamples[target_level]

                bg_image = slide.read_region((0, 0), target_level, (w_l, h_l)).convert('RGB')
                bg_image_np = np.array(bg_image)

                heatmap = np.zeros((h_l, w_l), dtype=np.float32)
                patch_size_l = int(Config.PATCH_SIZE / downsample)
                patch_size_l = max(1, patch_size_l)

                for (x_l0, y_l0), weight in zip(valid_coords, slice_A):
                    x_l = int(x_l0 / downsample)
                    y_l = int(y_l0 / downsample)
                    y_end = min(y_l + patch_size_l, h_l)
                    x_end = min(x_l + patch_size_l, w_l)
                    heatmap[y_l:y_end, x_l:x_end] = weight

                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()

                try:
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Serif CJK JP', 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                except Exception:
                    pass

                heatmap_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                
                mask = (heatmap > 0)[..., None]
                overlay = np.where(mask, np.uint8(0.6 * bg_image_np + 0.4 * heatmap_rgb), bg_image_np)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(bg_image_np)
                axes[0].set_title("Original WSI", fontsize=14)
                axes[0].axis('off')

                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title(f"Attention (class: {pred_name})", fontsize=14)
                axes[1].axis('off')

                axes[2].imshow(overlay)
                axes[2].set_title("Overlay", fontsize=14)
                axes[2].axis('off')

                svs_stem = Path(sp).stem
                # plt.suptitle(f"Sample: {sample_id}  |  WSI: {svs_stem}", fontsize=16)
                plt.tight_layout()

                # 按照用户要求，保存在对应的 processed 目录下
                # 比如 save_dir 是 task9_heatmap_wsi/Fibroma/冯庚辉
                # 我们需要保存在 task9_heatmap_wsi/Fibroma/冯庚辉/processed/冯庚辉-中.png
                final_save_dir = save_dir / "processed"
                final_save_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{svs_stem}.png"
                plt.savefig(str(final_save_dir / fname), dpi=150, bbox_inches='tight')
                plt.close(fig)

                # 保留拼接图；同级以 stem 为名的文件夹内存三张原始分辨率大图
                sub_dir = final_save_dir / svs_stem
                sub_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sub_dir / "original.png"), cv2.cvtColor(bg_image_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sub_dir / "heatmap.png"), heatmap_bgr)
                cv2.imwrite(str(sub_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        finally:
            hook_handle.remove()
            self._attention_weights = None
            self.model.zero_grad()


def main_heatmap_wsi(test_root, results_dir):
    """提取全模态全开的情况下，WSI分支的特征热图并绘制"""
    _configure_for_7cls() 
    
    print("\n" + "#" * 70)
    print("#  Task: Heatmap Extraction (WSI Branch)")
    print("#  Test dataset: " + test_root)
    print("#  Results dir : " + results_dir)
    print("#" * 70)

    pipeline = _build_pipeline(Config.CHECKPOINT_DIR)
    visualizer = WSIHeatmapVisualizer(pipeline)

    use_txt, use_img, use_svs = True, True, True
    samples = _collect_samples(test_root, use_txt, use_img, use_svs)
    print(f"  Found {len(samples)} test samples.")
    if len(samples) == 0:
        return

    base_dir = os.path.dirname(test_root)

    for sample in tqdm(samples, desc="[Heatmap-WSI]"):
        try:
            sample_rel_dir = os.path.relpath(sample["path"], base_dir)
            save_dir = os.path.join(results_dir, sample_rel_dir)
            
            sample_id = os.path.basename(sample["path"])
            
            visualizer(
                text_input=sample["text"],
                image_paths=sample["images"],
                svs_paths=sample["svs"],
                save_dir=Path(save_dir),
                sample_id=sample_id,
            )
        except Exception as e:
            print(f"  Error on {sample['path']}: {e}")

    print("\n>>> WSI Heatmap extraction finished. Saved to:", results_dir)


if __name__ == "__main__":
    test_root = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets-8csl-70_30-new/test"
    test_root = "/media/codingma/LLM/data-20260220/train-val/val-10"
    results_dir = "/media/codingma/LLM/results-testset/heatmap-wsi-test"
    if len(sys.argv) > 1:
        task = sys.argv[1]
        if task == "3cls":
            data_dict = [
                ("/media/codingma/LLM/data-20260220/train-val/train-60",
                "/media/codingma/LLM/results4/3cls-train-60"),
                ("/media/codingma/LLM/data-20260220/train-val/val-10",
                "/media/codingma/LLM/results4/3cls-val-10"),
                ("/media/codingma/LLM/lcx/Medical_Info_Classification/datasets-8csl-70_30-new/test",
                "/media/codingma/LLM/results4/3cls"),
                ("/media/codingma/LLM/data-20260220/internal-val",
                "/media/codingma/LLM/results4/3cls-internal-val"),
                ("/media/codingma/LLM/data-20260220/external-val",
                "/media/codingma/LLM/results4/3cls-external-val")
            ]
            for input_data, output_dir in data_dict:
                print(f"\n\n=== Evaluating on dataset: {input_data} ===")
                main_3cls(test_root=input_data, results_dir=output_dir)

        elif task == "7cls":
            data_dict = [
                ("/media/codingma/LLM/data-20260220/train-val/train-60",
                "/media/codingma/LLM/results4/8cls-train-60"),
                ("/media/codingma/LLM/data-20260220/train-val/val-10",
                "/media/codingma/LLM/results4/8cls-val-10"),
                ("/media/codingma/LLM/lcx/Medical_Info_Classification/datasets-8csl-70_30-new/test",
                "/media/codingma/LLM/results4/8cls"),
                ("/media/codingma/LLM/data-20260220/internal-val",
                "/media/codingma/LLM/results4/8cls-internal-val"),
                ("/media/codingma/LLM/data-20260220/external-val",
                "/media/codingma/LLM/results4/8cls-external-val")
            ]
            
            for input_data, output_dir in data_dict:
                print(f"\n\n=== Evaluating on dataset: {input_data} ===")
                main_7cls(test_root=input_data, results_dir=output_dir)
            
        elif task == "heatmap":
            results_dir = "/media/codingma/LLM/results-testset/heatmap-test-v2"
            main_heatmap(test_root=test_root, results_dir=results_dir)
        elif task == "heatmap-wsi":
            results_dir = "/media/codingma/LLM/results-testset/heatmap-wsi-test-v2"
            main_heatmap_wsi(test_root=test_root, results_dir=results_dir)
