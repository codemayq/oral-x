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

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    多标签分类的交叉熵 (ZLPR Loss)
    Reference: https://spaces.ac.cn/archives/7359
    y_true: multi-hot vector (0 or 1)
    y_pred: logits (before sigmoid/softmax)
    """
    # 调整 y_pred，使得正例 > 0，负例 < 0
    y_pred = (1 - 2 * y_true) * y_pred
    
    # 构造 log(1 + sum(e^neg)) + log(1 + sum(e^pos))
    # 使用 1e12 这种大数来 mask 掉不需要的部分
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    
    zeros = torch.zeros_like(y_pred[..., :1])
    
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    
    return neg_loss + pos_loss

def _append_epoch_losses_csv(csv_path: str, epoch: int, train_loss: float, val_loss: float) -> None:
    """
    每个 epoch 结束就追加一行到 CSV，避免等训练结束才落盘。
    CSV columns: epoch,train_loss,val_loss
    """
    file_exists = os.path.exists(csv_path)
    # newline="" 避免 Windows 下空行；linux 下也安全
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
    """
    按 step 频繁验证时的日志（每次验证追加一行并 flush）。
    CSV columns: epoch,global_step,train_loss_window,val_loss,val_acc
    """
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

def _quick_validate(model, val_loader, device):
    """
    轻量验证：只计算 val_loss 与多标签 subset accuracy（不生成 report，不保存全量预测）。
    """
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

            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                loss = torch.mean(multilabel_categorical_crossentropy(logits.float(), labels.float()))

            val_loss_sum += float(loss.item())

            preds = (logits > 0).float()
            # subset accuracy: 每个样本所有标签完全一致才算对
            batch_correct = (preds == labels).all(dim=1).sum().item()
            correct_samples += int(batch_correct)
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

def _safe_macro_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim == 1:
        return float(roc_auc_score(y_true, y_prob))
    per_class = []
    for i in range(y_true.shape[1]):
        if np.unique(y_true[:, i]).size < 2:
            continue
        per_class.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    if not per_class:
        return float("nan")
    return float(np.mean(per_class))

def _safe_micro_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim == 1:
        return float(roc_auc_score(y_true, y_prob))
    try:
        return float(roc_auc_score(y_true, y_prob, average="micro"))
    except Exception:
        return float("nan")

def _auc_diagnostics(y_true, y_prob, class_names=None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_prob = y_prob.reshape(-1, 1)
    n_classes = y_true.shape[1]
    names = class_names or [f"class_{i}" for i in range(n_classes)]
    per_class = []
    lines = []
    for i in range(n_classes):
        y_i = y_true[:, i]
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
    micro = _safe_micro_auc(y_true, y_prob)
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
    # 1. 准备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 确保保存目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 2. 数据
    print("Initializing Dataset...")
    print(f"Data Split (run={run_tag}): Train {len(train_df)} | Val {len(val_df)}")
    
    # 训练集：开启 expand_factor
    train_dataset = MultimodalDataset(
        train_df, 
        mode='train',
        expand_factor=Config.DATA_EXPAND_FACTOR,
        use_txt=Config.USE_TXT,
        use_img=Config.USE_IMG,
        use_svs=Config.USE_SVS
    )
    # 验证集：不扩充
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

    # 3. 模型
    print("Initializing Model (This may take time loading Qwen)...")
    model = UnifiedMultimodalModel().to(device)

    # 4. 优化器 & Loss
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    # [修改] 使用自定义 Multilabel Loss，不需要实例化标准 Loss
    # criterion = nn.CrossEntropyLoss()
    
    # 混合精度 Scaler
    scaler = torch.amp.GradScaler(device=device.type, enabled=Config.USE_AMP)

    start_epoch = 0
    best_val_f1 = 0.0
    global_step = 0

    max_steps = int(getattr(Config, "MAX_STEPS", 0))
    
    # ==================== 加载 Checkpoint (如果存在) ====================
    # 断点续训用 last.pth（目录模式）；Previous best Val F1 仅从 best_val.pth 读取，与 last 无关。
    checkpoint_path = Config.CHECKPOINT_DIR
    if checkpoint_path:
        if os.path.isdir(checkpoint_path):
            resume_ckpt = os.path.join(checkpoint_path, "last.pth")
            best_ckpt = os.path.join(checkpoint_path, "best_val.pth")
        else:
            resume_ckpt = checkpoint_path
            best_ckpt = os.path.join(os.path.dirname(checkpoint_path) or ".", "best_val.pth")

        if os.path.exists(best_ckpt):
            best_meta = None
            try:
                # 仅读标量 val_f1：整包在 CPU 上载入，取完即释放，避免 GPU 上多占一整份 best 权重导致 OOM
                best_meta = torch.load(best_ckpt, map_location="cpu")
                if isinstance(best_meta, dict) and "val_f1" in best_meta:
                    best_val_f1 = float(best_meta["val_f1"])
                    print(f"[Train] Previous best Val F1 (from best_val.pth): {best_val_f1:.4f}")
            except Exception as e:
                print(f"[Train] Warning: Failed to read val_f1 from {best_ckpt}: {e}")
            finally:
                if best_meta is not None:
                    del best_meta
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if os.path.exists(resume_ckpt):
            print(f"[Train] Resuming training from {resume_ckpt}...")
            checkpoint = torch.load(resume_ckpt, map_location=device)

            # 兼容不同的 key
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # 移除可能的 module. 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            print("[Train] Model weights loaded.")

            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                print(f"[Train] Resuming from epoch {start_epoch}")

            if "optimizer" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    print("[Train] Optimizer state loaded.")
                except Exception as e:
                    print(f"[Train] Warning: Failed to load optimizer state: {e}")

            if "global_step" in checkpoint:
                global_step = int(checkpoint["global_step"])
                print(f"[Train] Resumed global_step: {global_step}")
        elif os.path.isdir(checkpoint_path):
            print(
                f"[Train] Warning: last checkpoint {resume_ckpt} not found. "
                "Starting model/optimizer from scratch."
            )
        else:
            print(
                f"[Train] Warning: Checkpoint {resume_ckpt} not found. Starting from scratch."
            )
    
    
    total_steps_per_epoch = len(train_loader)
    loss_csv_path = os.path.join(checkpoint_dir, "loss_log.csv")
    eval_csv_path = os.path.join(checkpoint_dir, "eval_log.csv")
    auc_csv_path = os.path.join(checkpoint_dir, "auc_log.csv")
    eval_every_steps = int(getattr(Config, "EVAL_EVERY_STEPS", 100))

    if max_steps > 0:
        effective_epochs = start_epoch + 1000000
        print(f"[Train] Step-based training: max_steps={max_steps}, eval_every_steps={eval_every_steps}")
    else:
        effective_epochs = Config.EPOCHS

    # 5. 循环
    for epoch in range(start_epoch, effective_epochs):
        if max_steps > 0 and global_step >= max_steps:
            print(f"[Train] Reached max_steps={max_steps}, training finished.")
            break

        # ================= Training =================
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        train_probs = []

        optimizer.zero_grad() # 梯度清零
        # 用于"每隔 N 步验证一次"的训练 loss 窗口统计
        train_loss_window = 0.0
        train_loss_window_n = 0
        
        print(f"\n***** Epoch {epoch+1}/{effective_epochs} Start *****")
        
        if max_steps > 0:
            desc = f"Step {global_step}/{max_steps}"
        else:
            desc = f"Epoch {epoch+1}"
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=desc)
        for i, batch in pbar:
            # 搬运数据
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            normal_imgs = batch['normal_imgs'].to(device)
            wsi_feat = batch['wsi_feat'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)
            labels = batch['labels'].to(device)

            # AMP Forward
            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                # [修改] 使用 Multi-label Loss
                # Loss 计算强制使用 float32 避免数值溢出
                loss = torch.mean(multilabel_categorical_crossentropy(logits.float(), labels.float()))
                loss = loss / Config.GRAD_ACCUM_STEPS

            # AMP Backward
            scaler.scale(loss).backward()
            
            grad_norm = 0.0
            
            # 梯度累积更新
            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                # 计算 Grad Norm (在 unscale 之前或之后都可以，通常在 unscale 后 clip_grad_norm 时计算)
                # 为了打印真实的 grad norm，我们先 unscale
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 也可以设为很大来只观测
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 统计
            current_loss = loss.item() * Config.GRAD_ACCUM_STEPS
            train_loss += current_loss
            train_loss_window += float(current_loss)
            train_loss_window_n += 1
            
            # [修改] 多标签预测: logits > 0 为正类
            preds = (logits > 0).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(torch.sigmoid(logits.float()).detach().cpu().numpy())

            global_step += 1
            if max_steps > 0:
                pbar.set_description(f"Step {global_step}/{max_steps}")
                if global_step >= max_steps:
                    break
            
            # 打印日志 (Step, Loss, Grad Norm)
            # {'loss': 0.1234, 'grad_norm': 0.5678, 'learning_rate': 1e-5, 'epoch': 1.01, 'step': 100}
            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                 log_msg = f"{{'loss': {current_loss:.4f}, 'grad_norm': {grad_norm:.4f}, 'epoch': {epoch + (i + 1) / total_steps_per_epoch:.2f}, 'step': {global_step}}}"
                 tqdm.write(log_msg)
                 pbar.set_postfix({'loss': f"{current_loss:.4f}"})

                 # ======= 每隔 N 个 optimizer step 做一次验证并保存 =======
                 if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                     # 计算窗口平均 train loss（从上次验证后累计到现在）
                     avg_train_loss_window = train_loss_window / max(1, train_loss_window_n)
                     # 清空窗口（下一次验证重新累计）
                     train_loss_window = 0.0
                     train_loss_window_n = 0

                     print(f"\n[Eval @ step {global_step}] Running quick validation...")
                     avg_val_loss_step, val_acc_step = _quick_validate(model, val_loader, device)

                     # 恢复训练模式
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
                         print(f"✅ Eval log appended to: {eval_csv_path}")
                     except Exception as e:
                         print(f"⚠️ Failed to append eval log to CSV: {e}")

                     # 释放缓存，避免频繁验证造成显存碎片
                     torch.cuda.empty_cache()
                     gc.collect()

        # [新增] Epoch 结束，清理显存和内存
        try:
            del logits, loss, preds, input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask, labels
        except NameError:
            pass # 可能 loop 一次都没进
        torch.cuda.empty_cache()
        gc.collect()

        # Train Metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_auc = _safe_macro_auc(train_labels, train_probs)
        train_macro_auc, train_micro_auc, train_auc_lines = _auc_diagnostics(
            train_labels, train_probs, Config.TARGET_CLASS_NAMES
        )
        avg_train_loss = train_loss / len(train_loader)
        
        # [新增] 释放列表以节省内存
        del train_preds, train_labels, train_probs
        gc.collect()
        
        # ================= Validation =================
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_probs = []
        
        print(f"\nRunning Validation...")
        
        # 验证阶段不需要梯度
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
                    # [修改] Val Loss
                    # Loss 计算强制使用 float32 避免数值溢出
                    loss = torch.mean(multilabel_categorical_crossentropy(logits.float(), labels.float()))
                
                val_loss += loss.item()
                # [修改] Val Preds
                preds = (logits > 0).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(torch.sigmoid(logits.float()).detach().cpu().numpy())
        
        # Val Metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_auc = _safe_macro_auc(val_labels, val_probs)
        avg_val_loss = val_loss / len(val_loader)
        
        # [新增] 打印详细的 Classification Report
        print("\n" + "="*30 + " Validation Report " + "="*30)
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
        print("="*80)
        
        print(f"Results Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
        print(f"  Train AUC macro: {train_macro_auc:.4f} | micro: {train_micro_auc:.4f}")
        print("  Train AUC per-class:")
        for line in train_auc_lines:
            print(line)
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        val_macro_auc, val_micro_auc, val_auc_lines = _auc_diagnostics(
            val_labels, val_probs, Config.TARGET_CLASS_NAMES
        )
        print(f"  Val AUC macro: {val_macro_auc:.4f} | micro: {val_micro_auc:.4f}")
        print("  Val AUC per-class:")
        for line in val_auc_lines:
            print(line)

        # [新增] 清理 Val 临时变量
        del val_preds, val_labels, val_probs
        torch.cuda.empty_cache()
        gc.collect()

        # [新增] 每个 epoch 结束立即追加保存 loss 到 CSV（不等训练结束）
        try:
            _append_epoch_losses_csv(
                csv_path=loss_csv_path,
                epoch=epoch + 1,
                train_loss=float(avg_train_loss),
                val_loss=float(avg_val_loss),
            )
            print(f"✅ Loss log appended to: {loss_csv_path}")
        except Exception as e:
            print(f"⚠️ Failed to append loss log to CSV: {e}")

        # [新增] AUC 日志
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
            print(f"✅ AUC log appended to: {auc_csv_path}")
        except Exception as e:
            print(f"⚠️ Failed to append AUC log to CSV: {e}")

        # [新增] epoch 结束也追加一条“验证日志”（便于和 step 验证放到同一表看趋势）
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
        
        # Save checkpoint
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1
        }
    
        # 保存 Best (根据 Val F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(checkpoint_dir, "best_val.pth")
            torch.save(state, best_path)
            print(f"🌟 New Best Model (Val F1: {best_val_f1:.4f}) Saved to {best_path}")
    
        # 保存 Last
        last_path = os.path.join(checkpoint_dir, "last.pth")
        torch.save(state, last_path)
        print(f"💾 Last Model Saved")
        print("-" * 50)

def train():
    # 1. 准备
    _set_seed(Config.SEED)
    
    # 2. 数据
    print("Initializing Dataset...")
    
    # [修改] 读取全量数据并划分
    full_df = pd.read_csv(Config.DATA_INDEX_PATH)
    # 打乱数据
    full_df = full_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    
    split_idx = int(len(full_df) * Config.TRAIN_VAL_SPLIT)
    train_df_full = full_df.iloc[:split_idx].reset_index(drop=True)
    val_df = full_df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Data Split: Train {len(train_df_full)} | Val {len(val_df)}")

    # 学习曲线模式：多训练集规模
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

    # 默认：单次训练
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
