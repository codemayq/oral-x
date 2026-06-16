"""
单标签多类别训练模块
自动识别 3 分类 / 8 分类任务（由 Config.NUM_CLASSES 决定）。
使用 CrossEntropyLoss + argmax，每个 epoch 结束输出完整分类指标。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import gc
import random
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

from .config import Config, map_label_for_3cls_training
from .dataset import MultimodalDataset, collate_fn
from .model import UnifiedMultimodalModel


class FocalLoss(nn.Module):
    """Focal Loss with class weights and label smoothing.

    Reduces loss for well-classified (easy) samples so the model
    focuses on hard, minority-class examples.
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_macro_auc(y_true_indices, y_prob, num_classes) -> float:
    """单标签 macro AUC: 先将 y_true 转为 one-hot 再逐类计算。"""
    y_true_indices = np.asarray(y_true_indices)
    y_prob = np.asarray(y_prob)
    y_true_oh = np.eye(num_classes)[y_true_indices]
    per_class = []
    for i in range(num_classes):
        if np.unique(y_true_oh[:, i]).size < 2:
            continue
        per_class.append(roc_auc_score(y_true_oh[:, i], y_prob[:, i]))
    if not per_class:
        return float("nan")
    return float(np.mean(per_class))


def _auc_diagnostics(y_true_indices, y_prob, num_classes, class_names=None):
    """逐类 AUC 诊断，返回 (macro, micro, 每类诊断字符串列表)。"""
    y_true_indices = np.asarray(y_true_indices)
    y_prob = np.asarray(y_prob)
    y_true_oh = np.eye(num_classes)[y_true_indices]
    names = class_names or [f"class_{i}" for i in range(num_classes)]
    per_class = []
    lines = []
    for i in range(num_classes):
        y_i = y_true_oh[:, i]
        p_i = y_prob[:, i]
        pos = int(y_i.sum())
        neg = int(len(y_i) - pos)
        if np.unique(y_i).size < 2:
            auc_i = float("nan")
        else:
            auc_i = float(roc_auc_score(y_i, p_i))
        per_class.append(auc_i)
        auc_str = f"{auc_i:.4f}" if np.isfinite(auc_i) else "nan"
        lines.append(f"  - {names[i]}: pos={pos}, neg={neg}, auc={auc_str}")
    macro = float(np.nanmean(per_class)) if np.isfinite(np.nanmean(per_class)) else float("nan")
    try:
        micro = float(roc_auc_score(y_true_oh, y_prob, average="micro"))
    except Exception:
        micro = float("nan")
    return macro, micro, lines


def _build_scheduler(optimizer, steps_per_epoch: int):
    """根据 Config 构建学习率调度器，返回 (scheduler, is_epoch_level)。"""
    if not getattr(Config, "USE_SCHEDULER", False):
        return None, True

    stype = getattr(Config, "SCHEDULER_TYPE", "cosine_warm_restarts")
    if stype == "cosine_warm_restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(Config, "COSINE_T0", 2) * steps_per_epoch,
            T_mult=getattr(Config, "COSINE_T_MULT", 2),
            eta_min=getattr(Config, "COSINE_ETA_MIN", 1e-7),
        )
        return scheduler, False  # step-level
    elif stype == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(Config, "PLATEAU_FACTOR", 0.5),
            patience=getattr(Config, "PLATEAU_PATIENCE", 2),
        )
        return scheduler, True  # epoch-level, needs val_loss
    else:
        return None, True


def _warmup_lr(optimizer, epoch: int, step: int, warmup_steps: int, base_lr: float):
    """线性 warmup：在 warmup 阶段按 step 线性增大学习率。"""
    if warmup_steps <= 0 or step >= warmup_steps:
        return
    lr = base_lr * (step + 1) / warmup_steps
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train():
    _set_seed(Config.SEED)

    num_classes = Config.NUM_CLASSES
    class_names = Config.TARGET_CLASS_NAMES
    print(f"Task: {num_classes}-class single-label classification")
    print(f"Classes: {class_names}")

    # ==================== 数据 ====================
    full_df = pd.read_csv(Config.DATA_INDEX_PATH)
    full_df = full_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    split_idx = int(len(full_df) * Config.TRAIN_VAL_SPLIT)
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    val_df = full_df.iloc[split_idx:].reset_index(drop=True)
    print(f"Data Split: Train {len(train_df)} | Val {len(val_df)}")

    train_dataset = MultimodalDataset(
        train_df,
        mode="train",
        expand_factor=Config.DATA_EXPAND_FACTOR,
        use_txt=Config.USE_TXT,
        use_img=Config.USE_IMG,
        use_svs=Config.USE_SVS,
    )
    val_dataset = MultimodalDataset(
        val_df,
        mode="val",
        use_txt=Config.USE_TXT,
        use_img=Config.USE_IMG,
        use_svs=Config.USE_SVS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ==================== 模型 ====================
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    print("Initializing Model (This may take time loading Qwen)...")
    model = UnifiedMultimodalModel().to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # ---- 类别加权 Focal Loss + label smoothing ----
    label_counts = Counter(map_label_for_3cls_training(x) for x in train_df["label"].tolist())
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_samples / (num_classes * max(label_counts.get(i, 1), 1))
         for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    print(f"Class weights: {dict(zip(class_names, class_weights.cpu().tolist()))}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)

    # Qwen 以 float16 加载 -> 梯度也是 FP16 -> GradScaler.unscale_ 会报错
    # 使用 bfloat16 autocast + 禁用 GradScaler 来规避
    use_bf16 = Config.USE_AMP and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = Config.USE_AMP and not use_bf16
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_scaler)

    steps_per_epoch = len(train_loader) // max(1, Config.GRAD_ACCUM_STEPS)
    scheduler, scheduler_epoch_level = _build_scheduler(optimizer, steps_per_epoch)

    warmup_epochs = int(getattr(Config, "WARMUP_EPOCHS", 0))
    warmup_steps = warmup_epochs * steps_per_epoch
    base_lr = Config.LEARNING_RATE

    best_val_f1 = 0.0
    global_step = 0
    max_steps = int(getattr(Config, "MAX_STEPS", 0))
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    if max_steps > 0:
        effective_epochs = 1000000
        print(f"[Train] Step-based training: max_steps={max_steps}")
    else:
        effective_epochs = Config.EPOCHS

    # ==================== 训练循环 ====================
    for epoch in range(effective_epochs):
        if max_steps > 0 and global_step >= max_steps:
            print(f"[Train] Reached max_steps={max_steps}, training finished.")
            break
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []
        train_probs = []
        global_step_in_epoch = 0

        optimizer.zero_grad()

        print(f"\n{'='*25} Epoch {epoch+1}/{effective_epochs} {'='*25}")

        if max_steps > 0:
            desc = f"Step {global_step}/{max_steps}"
        else:
            desc = f"Epoch {epoch+1}"
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=desc)
        for i, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            normal_imgs = batch["normal_imgs"].to(device)
            wsi_feat = batch["wsi_feat"].to(device)
            wsi_mask = batch["wsi_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP, dtype=amp_dtype):
                logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                loss = criterion(logits.float(), labels)
                loss = loss / Config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            grad_norm = 0.0
            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                abs_step = epoch * steps_per_epoch + global_step_in_epoch
                _warmup_lr(optimizer, epoch, abs_step, warmup_steps, base_lr)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step_in_epoch += 1

                # step-level scheduler（warmup 结束后才启用）
                if scheduler and not scheduler_epoch_level and abs_step >= warmup_steps:
                    scheduler.step()

            current_loss = loss.item() * Config.GRAD_ACCUM_STEPS
            train_loss += current_loss

            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            train_probs.extend(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())

            global_step += 1
            if max_steps > 0:
                pbar.set_description(f"Step {global_step}/{max_steps}")
                if global_step >= max_steps:
                    break

            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log_msg = (
                    f"{{'loss': {current_loss:.4f}, 'grad_norm': {grad_norm:.4f}, "
                    f"'lr': {lr_now:.2e}, 'epoch': {epoch + (i + 1) / len(train_loader):.2f}, "
                    f"'step': {global_step}}}"
                )
                tqdm.write(log_msg)
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        # 释放训练 batch 变量
        try:
            del logits, loss, preds, input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask, labels
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        # ==================== 训练指标 ====================
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels_list, train_preds)
        train_auc = _safe_macro_auc(train_labels_list, train_probs, num_classes)

        del train_preds, train_labels_list, train_probs
        gc.collect()

        # ==================== 验证 ====================
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []
        val_probs = []

        print("Running Validation...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                normal_imgs = batch["normal_imgs"].to(device)
                wsi_feat = batch["wsi_feat"].to(device)
                wsi_mask = batch["wsi_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP, dtype=amp_dtype):
                    logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                    loss = criterion(logits.float(), labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                val_probs.extend(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds, average="macro", zero_division=0)
        val_auc = _safe_macro_auc(val_labels_list, val_probs, num_classes)

        # ==================== 输出指标 ====================
        print(f"\n{'='*30} Epoch {epoch+1} Results {'='*30}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC(macro): {train_auc:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1(macro): {val_f1:.4f} | Val AUC(macro): {val_auc:.4f}")

        # classification_report
        print(f"\n{'='*30} Classification Report {'='*30}")
        try:
            report = classification_report(
                val_labels_list,
                val_preds,
                target_names=class_names,
                zero_division=0,
                digits=4,
            )
            print(report)
        except Exception as e:
            print(f"Error generating classification report: {e}")

        # per-class AUC
        val_macro_auc, val_micro_auc, val_auc_lines = _auc_diagnostics(
            val_labels_list, val_probs, num_classes, class_names
        )
        print(f"  Val AUC  macro: {val_macro_auc:.4f} | micro: {val_micro_auc:.4f}")
        print("  Val AUC per-class:")
        for line in val_auc_lines:
            print(line)
        print("=" * 80)

        # epoch-level scheduler
        if scheduler and scheduler_epoch_level:
            scheduler.step(avg_val_loss)

        del val_preds, val_labels_list, val_probs
        torch.cuda.empty_cache()
        gc.collect()

        # ==================== checkpoint ====================
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_f1": val_f1,
        }

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(Config.CHECKPOINT_DIR, "best_val.pth")
            torch.save(state, best_path)
            print(f"New Best Model (Val macro-F1: {best_val_f1:.4f}, Acc: {val_acc:.4f}) saved to {best_path}")

        last_path = os.path.join(Config.CHECKPOINT_DIR, "last.pth")
        torch.save(state, last_path)
        print(f"Last Model saved to {last_path}")
        print("-" * 50)

    print(f"\nTraining finished. Best Val macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()
