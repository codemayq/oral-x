"""
单标签多类别训练模块 (8 类)
使用 CrossEntropyLoss + argmax，适用于 Config.MULTI_LABEL = False 的场景。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
import csv
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

from .config import Config
from .dataset import MultimodalDataset, collate_fn
from .model import UnifiedMultimodalModel


def _append_epoch_losses_csv(csv_path: str, epoch: int, train_loss: float, val_loss: float) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}"])
        f.flush()


def _append_eval_metrics_csv(
    csv_path: str,
    epoch: int,
    global_step: int,
    train_loss_window: float,
    val_loss: float,
    val_acc: float,
) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            writer.writerow(["epoch", "global_step", "train_loss_window", "val_loss", "val_acc"])
        writer.writerow(
            [
                int(epoch),
                int(global_step),
                f"{train_loss_window:.8f}",
                f"{val_loss:.8f}",
                f"{val_acc:.8f}",
            ]
        )
        f.flush()


def _quick_validate(model, val_loader, criterion, device):
    model.eval()
    val_loss_sum = 0.0
    total_samples = 0
    correct_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            normal_imgs = batch['normal_imgs'].to(device)
            wsi_feat = batch['wsi_feat'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP, dtype=amp_dtype):
                logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                loss = criterion(logits.float(), labels)

            val_loss_sum += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            correct_samples += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))

    avg_val_loss = val_loss_sum / max(1, len(val_loader))
    val_acc = correct_samples / max(1, total_samples)
    return avg_val_loss, val_acc


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_macro_auc(y_true_indices, y_prob, num_classes) -> float:
    """单标签 AUC: 先将 y_true 转为 one-hot 再逐类计算。"""
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

def _append_auc_csv(
    csv_path: str,
    epoch: int,
    train_size: int,
    train_auc: float,
    val_auc: float,
    run_tag: str,
    seed: int,
) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            writer.writerow(["run_tag", "seed", "train_size", "epoch", "train_auc", "val_auc"])
        writer.writerow(
            [run_tag, int(seed), int(train_size), int(epoch), f"{train_auc:.8f}", f"{val_auc:.8f}"]
        )
        f.flush()


def _train_with_dfs(train_df, val_df, checkpoint_dir: str, run_tag: str, seed: int, train_size: int):
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Initializing Dataset...")
    print(f"Data Split (run={run_tag}): Train {len(train_df)} | Val {len(val_df)}")

    train_dataset = MultimodalDataset(
        train_df,
        mode='train',
        expand_factor=Config.DATA_EXPAND_FACTOR,
        use_txt=Config.USE_TXT,
        use_img=Config.USE_IMG,
        use_svs=Config.USE_SVS
    )
    val_dataset = MultimodalDataset(
        val_df,
        mode='val',
        use_txt=Config.USE_TXT,
        use_img=Config.USE_IMG,
        use_svs=Config.USE_SVS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("Initializing Model (This may take time loading Qwen)...")
    model = UnifiedMultimodalModel().to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiMarginLoss(p=1, margin=1.0, weight=None,  reduction='mean')

    scaler = torch.amp.GradScaler(device=device.type, enabled=Config.USE_AMP)

    best_val_acc = 0.0
    required_acc_least = 0.6

    total_steps_per_epoch = len(train_loader)
    loss_csv_path = os.path.join(checkpoint_dir, "loss_log.csv")
    eval_csv_path = os.path.join(checkpoint_dir, "eval_log.csv")
    auc_csv_path = os.path.join(checkpoint_dir, "auc_log.csv")
    eval_every_steps = int(getattr(Config, "EVAL_EVERY_STEPS", 100))

    num_classes = Config.NUM_CLASSES

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        train_probs = []

        global_step = 0
        optimizer.zero_grad()
        train_loss_window = 0.0
        train_loss_window_n = 0

        print(f"\n***** Epoch {epoch+1}/{Config.EPOCHS} Start *****")

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            normal_imgs = batch['normal_imgs'].to(device)
            wsi_feat = batch['wsi_feat'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                loss = criterion(logits.float(), labels)
                loss = loss / Config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            grad_norm = 0.0
            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            current_loss = loss.item() * Config.GRAD_ACCUM_STEPS
            train_loss += current_loss
            train_loss_window += float(current_loss)
            train_loss_window_n += 1

            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())

            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                log_msg = f"{{'loss': {current_loss:.4f}, 'grad_norm': {grad_norm:.4f}, 'epoch': {epoch + (i + 1) / total_steps_per_epoch:.2f}, 'step': {global_step}}}"
                tqdm.write(log_msg)
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

                if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                    avg_train_loss_window = train_loss_window / max(1, train_loss_window_n)
                    train_loss_window = 0.0
                    train_loss_window_n = 0

                    print(f"\n[Eval @ step {global_step}] Running quick validation...")
                    avg_val_loss_step, val_acc_step = _quick_validate(model, val_loader, criterion, device)
                    model.train()

                    try:
                        _append_eval_metrics_csv(
                            csv_path=eval_csv_path,
                            epoch=epoch + 1,
                            global_step=global_step,
                            train_loss_window=float(avg_train_loss_window),
                            val_loss=float(avg_val_loss_step),
                            val_acc=float(val_acc_step),
                        )
                    except Exception as e:
                        print(f"Failed to append eval log to CSV: {e}")

                    torch.cuda.empty_cache()
                    gc.collect()

        try:
            del logits, loss, preds, input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask, labels
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        # Train Metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_auc = _safe_macro_auc(train_labels, train_probs, num_classes)
        train_macro_auc, train_micro_auc, train_auc_lines = _auc_diagnostics(
            train_labels, train_probs, num_classes, Config.TARGET_CLASS_NAMES
        )
        avg_train_loss = train_loss / len(train_loader)

        del train_preds, train_labels, train_probs
        gc.collect()

        # ================= Validation =================
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_probs = []

        print(f"\nRunning Validation...")

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                normal_imgs = batch['normal_imgs'].to(device)
                wsi_feat = batch['wsi_feat'].to(device)
                wsi_mask = batch['wsi_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                    logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                    loss = criterion(logits.float(), labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_auc = _safe_macro_auc(val_labels, val_probs, num_classes)
        avg_val_loss = val_loss / len(val_loader)

        print("\n" + "=" * 30 + " Validation Report " + "=" * 30)
        try:
            report = classification_report(
                val_labels,
                val_preds,
                target_names=Config.TARGET_CLASS_NAMES,
                zero_division=0,
                digits=4
            )
            print(report)
        except Exception as e:
            print(f"Error generating classification report: {e}")
        print("=" * 80)

        print(f"Results Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
        print(f"  Train AUC macro: {train_macro_auc:.4f} | micro: {train_micro_auc:.4f}")
        print("  Train AUC per-class:")
        for line in train_auc_lines:
            print(line)
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        val_macro_auc, val_micro_auc, val_auc_lines = _auc_diagnostics(
            val_labels, val_probs, num_classes, Config.TARGET_CLASS_NAMES
        )
        print(f"  Val AUC macro: {val_macro_auc:.4f} | micro: {val_micro_auc:.4f}")
        print("  Val AUC per-class:")
        for line in val_auc_lines:
            print(line)

        del val_preds, val_labels, val_probs
        torch.cuda.empty_cache()
        gc.collect()

        try:
            _append_epoch_losses_csv(
                csv_path=loss_csv_path,
                epoch=epoch + 1,
                train_loss=float(avg_train_loss),
                val_loss=float(avg_val_loss),
            )
        except Exception as e:
            print(f"Failed to append loss log to CSV: {e}")

        try:
            _append_auc_csv(
                csv_path=auc_csv_path,
                epoch=epoch + 1,
                train_size=train_size,
                train_auc=float(train_auc),
                val_auc=float(val_auc),
                run_tag=run_tag,
                seed=seed,
            )
        except Exception as e:
            print(f"Failed to append AUC log to CSV: {e}")

        try:
            _append_eval_metrics_csv(
                csv_path=eval_csv_path,
                epoch=epoch + 1,
                global_step=(epoch + 1) * max(1, len(train_loader) // max(1, Config.GRAD_ACCUM_STEPS)),
                train_loss_window=float(avg_train_loss),
                val_loss=float(avg_val_loss),
                val_acc=float(val_acc),
            )
        except Exception:
            pass

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1
        }

        if val_acc > max(best_val_acc, required_acc_least):
            best_val_acc = val_acc
            best_path = os.path.join(checkpoint_dir, "best_val.pth")
            torch.save(state, best_path)
            print(f"New Best Model (Val Acc: {best_val_acc:.4f}) Saved to {best_path}")

        last_path = os.path.join(checkpoint_dir, "last.pth")
        torch.save(state, last_path)
        print(f"Last Model Saved")
        print("-" * 50)


def train():
    _set_seed(Config.SEED)

    print("Initializing Dataset (Single-Label Mode, 8 classes)...")

    full_df = pd.read_csv(Config.DATA_INDEX_PATH)
    full_df = full_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    split_idx = int(len(full_df) * Config.TRAIN_VAL_SPLIT)
    train_df_full = full_df.iloc[:split_idx].reset_index(drop=True)
    val_df = full_df.iloc[split_idx:].reset_index(drop=True)

    print(f"Data Split: Train {len(train_df_full)} | Val {len(val_df)}")

    if getattr(Config, "LEARNING_CURVE_ENABLE", False):
        sizes = list(getattr(Config, "LEARNING_CURVE_TRAIN_SIZES", [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]))
        repeats = int(getattr(Config, "LEARNING_CURVE_REPEATS", 1))
        base_seed = int(getattr(Config, "LEARNING_CURVE_BASE_SEED", Config.SEED))
        root_dir = os.path.join(Config.CHECKPOINT_DIR, "learning_curve")
        os.makedirs(root_dir, exist_ok=True)

        for size in sizes:
            for repeat in range(repeats):
                seed = base_seed + repeat
                _set_seed(seed)
                if size <= 1:
                    n_train = int(len(train_df_full) * float(size))
                else:
                    n_train = int(size)
                n_train = max(1, min(n_train, len(train_df_full)))

                train_df = train_df_full.sample(n=n_train, random_state=seed).reset_index(drop=True)
                run_tag = f"size_{n_train}_seed_{seed}"
                checkpoint_dir = os.path.join(root_dir, run_tag)

                print(f"\n===== Learning Curve Run: {run_tag} =====")
                _train_with_dfs(
                    train_df=train_df,
                    val_df=val_df,
                    checkpoint_dir=checkpoint_dir,
                    run_tag=run_tag,
                    seed=seed,
                    train_size=n_train,
                )
        return

    _train_with_dfs(
        train_df=train_df_full,
        val_df=val_df,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        run_tag="full",
        seed=Config.SEED,
        train_size=len(train_df_full),
    )


if __name__ == "__main__":
    train()
