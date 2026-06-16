"""
MoE 增强多模态模型训练模块 v2
适配 DeepSeek 风格 SparseMoE Transformer + EMA。

核心改进：
- WeightedRandomSampler 保证稀有类采样频率
- EMA 指数移动平均稳定验证表现
- 联合损失: Focal + 层次化 + 原型 + MoE 辅助损失
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import gc
import random
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

from .config import Config, map_label_for_3cls_training
from .dataset import MultimodalDataset, collate_fn
from .moe_model import MoEMultimodalModel, MoEFocalLoss, ModelEMA


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_macro_auc(y_true_indices, y_prob, num_classes) -> float:
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
        return scheduler, False
    elif stype == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(Config, "PLATEAU_FACTOR", 0.5),
            patience=getattr(Config, "PLATEAU_PATIENCE", 2),
        )
        return scheduler, True
    else:
        return None, True


def _warmup_lr(optimizer, epoch: int, step: int, warmup_steps: int, base_lr: float):
    if warmup_steps <= 0 or step >= warmup_steps:
        return
    lr = base_lr * (step + 1) / warmup_steps
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _build_weighted_sampler(train_df: pd.DataFrame, expand_factor: int):
    labels = [map_label_for_3cls_training(x) for x in train_df["label"].tolist()]
    if expand_factor > 1:
        labels = labels * expand_factor

    label_counts = Counter(labels)
    total = len(labels)
    class_weight = {cls: total / count for cls, count in label_counts.items()}
    sample_weights = [class_weight[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def train():
    _set_seed(Config.SEED)
    checkpoint_path = Config.CHECKPOINT_DIR
    num_classes = Config.NUM_CLASSES
    class_names = Config.TARGET_CLASS_NAMES
    print(f"[MoE-v2 Train] Task: {num_classes}-class single-label classification")
    print(f"[MoE-v2 Train] Classes: {class_names}")

    # ==================== 数据 ====================
    full_df = pd.read_csv(Config.DATA_INDEX_PATH)
    full_df = full_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    split_idx = int(len(full_df) * Config.TRAIN_VAL_SPLIT)
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    val_df = full_df.iloc[split_idx:].reset_index(drop=True)
    print(f"[MoE-v2 Train] Data Split: Train {len(train_df)} | Val {len(val_df)}")

    label_counts = Counter(map_label_for_3cls_training(x) for x in train_df["label"].tolist())
    total_samples = sum(label_counts.values())
    class_counts_list = [max(label_counts.get(i, 1), 1) for i in range(num_classes)]
    print(f"[MoE-v2 Train] Train distribution: {dict(zip(class_names, class_counts_list))}")

    train_dataset = MultimodalDataset(
        train_df, mode="train", expand_factor=Config.DATA_EXPAND_FACTOR,
        use_txt=Config.USE_TXT, use_img=Config.USE_IMG, use_svs=Config.USE_SVS,
    )
    val_dataset = MultimodalDataset(
        val_df, mode="val",
        use_txt=Config.USE_TXT, use_img=Config.USE_IMG, use_svs=Config.USE_SVS,
    )

    train_sampler = _build_weighted_sampler(train_df, Config.DATA_EXPAND_FACTOR)
    print("[MoE-v2 Train] Using WeightedRandomSampler")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        sampler=train_sampler, num_workers=4,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True,
    )

    # ==================== 模型 ====================
    device = torch.device(Config.DEVICE)
    print(f"[MoE-v2 Train] Using device: {device}")
    print("[MoE-v2 Train] Initializing MoE-v2 Model...")
    model = MoEMultimodalModel().to(device)

    start_epoch = 0
    best_val_f1 = 0.0

    # ==================== 加载 Checkpoint (如果存在) ====================
    if checkpoint_path:
        if os.path.isdir(checkpoint_path):
            ckpt_file = os.path.join(checkpoint_path, "best_val_moe.pth")
        else:
            ckpt_file = checkpoint_path

        if os.path.exists(ckpt_file):
            print(f"[MoE-v2 Train] Loading checkpoint from {ckpt_file}...")
            checkpoint = torch.load(ckpt_file, map_location=device)

            # 兼容不同的 key
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 移除可能的 module. 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            print("[MoE-v2 Train] Model weights loaded.")

            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"[MoE-v2 Train] Resuming from epoch {start_epoch}")
            
            if 'val_f1' in checkpoint:
                best_val_f1 = checkpoint['val_f1']
                print(f"[MoE-v2 Train] Previous best Val F1: {best_val_f1:.4f}")
        else:
            print(f"[MoE-v2 Train] Warning: Checkpoint {ckpt_file} not found. Starting from scratch.")

    # EMA (在加载权重后初始化，以同步 shadow weights)
    ema_decay = getattr(Config, "EMA_DECAY", 0.999)
    ema = ModelEMA(model, decay=ema_decay)
    print(f"[MoE-v2 Train] EMA enabled (decay={ema_decay})")

    # 参数分组: LayerNorm/bias/标量/原型 不做 weight_decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            # bias, LayerNorm weight/bias, 标量 gate, routed_scale 等
            no_decay_params.append(param)
        elif "norm" in name.lower():
            no_decay_params.append(param)
        elif "prototypes" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": Config.WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=Config.LEARNING_RATE)

    # 加载 Optimizer 状态
    if checkpoint_path and os.path.exists(ckpt_file) and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("[MoE-v2 Train] Optimizer state loaded.")
        except Exception as e:
            print(f"[MoE-v2 Train] Warning: Failed to load optimizer state: {e}")

    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"[MoE-v2 Train] Param groups: {n_decay:,} decay, {n_nodecay:,} no-decay")

    class_weights = torch.tensor(
        [total_samples / (num_classes * max(label_counts.get(i, 1), 1))
         for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    print(f"[MoE-v2 Train] Class weights: {dict(zip(class_names, class_weights.cpu().tolist()))}")

    criterion = MoEFocalLoss(
        alpha=class_weights, gamma=2.0, label_smoothing=0.1,
        lambda_coarse=0.3, lambda_proto=0.2, lambda_aux=0.01,
    ).to(device)

    use_bf16 = Config.USE_AMP and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = Config.USE_AMP and not use_bf16
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_scaler)

    steps_per_epoch = len(train_loader) // max(1, Config.GRAD_ACCUM_STEPS)
    scheduler, scheduler_epoch_level = _build_scheduler(optimizer, steps_per_epoch)

    warmup_epochs = int(getattr(Config, "WARMUP_EPOCHS", 0))
    warmup_steps = warmup_epochs * steps_per_epoch
    base_lr = Config.LEARNING_RATE

    best_val_f1 = max(best_val_f1, 0.0) # 确保非负
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # ==================== 训练循环 ====================
    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []
        train_probs = []
        global_step_in_epoch = 0

        optimizer.zero_grad()

        print(f"\n{'='*25} [MoE-v2] Epoch {epoch+1}/{Config.EPOCHS} {'='*25}")

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            normal_imgs = batch["normal_imgs"].to(device)
            wsi_feat = batch["wsi_feat"].to(device)
            wsi_mask = batch["wsi_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP, dtype=amp_dtype):
                model_output = model(
                    input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask,
                    labels=labels,
                )
                loss = criterion(model_output, labels)
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

                # EMA 更新
                ema.update(model)

                if scheduler and not scheduler_epoch_level and abs_step >= warmup_steps:
                    scheduler.step()

            current_loss = loss.item() * Config.GRAD_ACCUM_STEPS
            train_loss += current_loss

            logits = model_output["logits"].float()
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            train_probs.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log_msg = (
                    f"{{'loss': {current_loss:.4f}, 'grad_norm': {grad_norm:.4f}, "
                    f"'lr': {lr_now:.2e}, 'epoch': {epoch + (i + 1) / len(train_loader):.2f}, "
                    f"'step': {global_step_in_epoch}}}"
                )
                tqdm.write(log_msg)
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        try:
            del model_output, loss, preds, input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask, labels
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

        # ==================== 验证 (使用 EMA 参数) ====================
        ema.apply_shadow(model)
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []
        val_probs = []

        print("[MoE-v2 Train] Running Validation (EMA)...")
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
                    loss = F.cross_entropy(logits.float(), labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                val_probs.extend(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())

        ema.restore(model)

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds, average="macro", zero_division=0)
        val_auc = _safe_macro_auc(val_labels_list, val_probs, num_classes)

        # ==================== 输出指标 ====================
        print(f"\n{'='*30} [MoE-v2] Epoch {epoch+1} Results {'='*30}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC(macro): {train_auc:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1(macro): {val_f1:.4f} | Val AUC(macro): {val_auc:.4f}")

        print(f"\n{'='*30} Classification Report {'='*30}")
        try:
            report = classification_report(
                val_labels_list, val_preds,
                target_names=class_names, zero_division=0, digits=4,
            )
            print(report)
        except Exception as e:
            print(f"Error generating classification report: {e}")

        val_macro_auc, val_micro_auc, val_auc_lines = _auc_diagnostics(
            val_labels_list, val_probs, num_classes, class_names
        )
        print(f"  Val AUC  macro: {val_macro_auc:.4f} | micro: {val_micro_auc:.4f}")
        print("  Val AUC per-class:")
        for line in val_auc_lines:
            print(line)

        hier_w = torch.sigmoid(model.hier_gate).item()
        proto_w = torch.sigmoid(model.proto_gate).item()
        print(f"  Learnable gates: hier_weight={hier_w:.4f}, proto_weight={proto_w:.4f}")
        print("=" * 80)

        if scheduler and scheduler_epoch_level:
            scheduler.step(avg_val_loss)

        del val_preds, val_labels_list, val_probs
        torch.cuda.empty_cache()
        gc.collect()

        # ==================== checkpoint (保存 EMA 参数) ====================
        ema.apply_shadow(model)
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_f1": val_f1,
        }
        ema.restore(model)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(Config.CHECKPOINT_DIR, "best_val_moe.pth")
            torch.save(state, best_path)
            print(f"[MoE-v2] New Best (Val macro-F1: {best_val_f1:.4f}, Acc: {val_acc:.4f}) -> {best_path}")

        last_path = os.path.join(Config.CHECKPOINT_DIR, "last_moe.pth")
        torch.save(state, last_path)
        print(f"[MoE-v2] Last saved -> {last_path}")
        print("-" * 50)

    print(f"\n[MoE-v2 Train] Finished. Best Val macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()
