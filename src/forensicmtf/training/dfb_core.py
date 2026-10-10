from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import csv
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from forensicmtf.models.forensicmtf import ForensicMTF
from forensicmtf.training.core import (
    FF_SUBSETS,
    FocalLoss,
    ModelEMA,
    _prune_checkpoints,
    compute_auc,
    per_method_metrics,
    select_threshold,
)

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


@torch.no_grad()
def _heal_nonfinite_buffers(model: nn.Module, reference: nn.Module) -> int:
    """A non-finite forward pass (e.g. an extreme/degenerate input) can permanently
    poison BatchNorm running_mean/running_var, since those buffers update on every
    training-mode forward call regardless of whether backward/optimizer.step() runs -
    skipping the optimizer step alone does not protect against this. Once poisoned,
    the buffer's own EMA-style update (`(1-m)*NaN + m*x == NaN` for any x) never
    recovers on its own, and ModelEMA.update() would blend that permanent NaN into the
    EMA model on every subsequent step. Restore just the non-finite floating-point
    buffers/params from a known-good reference (the EMA model, which this function is
    only ever called before) so training can continue cleanly.
    """
    healed = 0
    model_sd = model.state_dict()
    ref_sd = reference.state_dict()
    for key, value in model_sd.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            value.copy_(ref_sd[key])
            healed += 1
    return healed


class MaskLoss(nn.Module):
    """BCE + soft-Dice on the upsampled localization logits, masked-mean over
    mask_valid samples only (returns 0 when no sample in the batch has a valid mask,
    e.g. an all-FaceShifter or all-Celeb-DF batch)."""

    def __init__(self, dice_smooth: float = 1.0):
        super().__init__()
        self.dice_smooth = dice_smooth

    def forward(self, mask_logits: torch.Tensor | None, mask_gt: torch.Tensor, mask_valid: torch.Tensor) -> torch.Tensor:
        if mask_logits is None or not mask_valid.any():
            device = mask_gt.device if mask_gt is not None else mask_logits.device
            return torch.tensor(0.0, device=device)
        logits_up = F.interpolate(mask_logits, size=mask_gt.shape[-2:], mode='bilinear', align_corners=False)
        bce = F.binary_cross_entropy_with_logits(logits_up, mask_gt, reduction='none').mean(dim=[1, 2, 3])
        probs = torch.sigmoid(logits_up)
        inter = (probs * mask_gt).sum(dim=[1, 2, 3])
        union = probs.sum(dim=[1, 2, 3]) + mask_gt.sum(dim=[1, 2, 3])
        dice = 1.0 - (2 * inter + self.dice_smooth) / (union + self.dice_smooth)
        per_sample = bce + dice
        return per_sample[mask_valid].mean()


def compute_mask_iou_per_sample(mask_logits: torch.Tensor | None, mask_gt: torch.Tensor, mask_valid: torch.Tensor,
                                threshold: float = 0.5) -> torch.Tensor | None:
    """Per-sample IoU for every mask_valid==True entry in the batch, in the same order
    they appear (i.e. index i of the returned tensor corresponds to the i-th True entry
    of mask_valid, matching mask_valid.nonzero()) - lets callers attribute IoU back to
    per-sample metadata (e.g. manipulation method) rather than only a batch-pooled mean."""
    if mask_logits is None or not mask_valid.any():
        return None
    logits_up = F.interpolate(mask_logits, size=mask_gt.shape[-2:], mode='bilinear', align_corners=False)
    pred = (torch.sigmoid(logits_up) > threshold).float()
    gt = (mask_gt > 0.5).float()
    inter = (pred * gt).sum(dim=[1, 2, 3])
    union = ((pred + gt) > 0).float().sum(dim=[1, 2, 3])
    iou = inter[mask_valid] / union[mask_valid].clamp(min=1e-6)
    empty = union[mask_valid] < 1e-6
    return torch.where(empty, torch.ones_like(iou), iou)


def compute_mask_iou(mask_logits: torch.Tensor | None, mask_gt: torch.Tensor, mask_valid: torch.Tensor,
                     threshold: float = 0.5) -> float:
    iou = compute_mask_iou_per_sample(mask_logits, mask_gt, mask_valid, threshold=threshold)
    if iou is None:
        return float('nan')
    return float(iou.mean().item())


def compute_mask_f1_per_sample(mask_logits: torch.Tensor | None, mask_gt: torch.Tensor, mask_valid: torch.Tensor,
                               threshold: float = 0.5) -> torch.Tensor | None:
    """Per-sample pixel-level F1 (Dice coefficient) for every mask_valid==True entry,
    same ordering convention as compute_mask_iou_per_sample. F1 = 2|pred (inter) gt| /
    (|pred|+|gt|), the standard segmentation-F1/Dice definition used by the baselines
    this is compared against in Table~\\ref{tab:v3_localization}."""
    if mask_logits is None or not mask_valid.any():
        return None
    logits_up = F.interpolate(mask_logits, size=mask_gt.shape[-2:], mode='bilinear', align_corners=False)
    pred = (torch.sigmoid(logits_up) > threshold).float()
    gt = (mask_gt > 0.5).float()
    inter = (pred * gt).sum(dim=[1, 2, 3])
    denom = pred.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3])
    f1 = 2 * inter[mask_valid] / denom[mask_valid].clamp(min=1e-6)
    empty = denom[mask_valid] < 1e-6
    return torch.where(empty, torch.ones_like(f1), f1)


class DFBMetricsLogger:
    """Per-epoch CSV logger for training/val metrics, including mask_iou/id_acc/id_f1/f1
    alongside the usual loss/acc/auc fields."""

    def __init__(self, path: str, metric_names=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        metric_names = metric_names or (['real'] + FF_SUBSETS)
        self.fields = [
            'epoch', 'train_loss', 'train_acc', 'train_auc', 'train_f1',
            'val_loss', 'val_acc', 'val_auc', 'val_f1', 'val_mask_iou', 'val_id_acc', 'val_id_f1', 'lr',
        ]
        for name in metric_names:
            self.fields.append(f'{name}_acc')
            self.fields.append(f'{name}_auc')
        if not self.path.exists():
            with open(self.path, 'w', newline='') as fh:
                csv.DictWriter(fh, fieldnames=self.fields).writeheader()

    def write(self, row: dict):
        with self._lock:
            with open(self.path, 'a', newline='') as fh:
                csv.DictWriter(fh, fieldnames=self.fields, extrasaction='ignore').writerow(row)


def run_epoch_dfb(model, loader, focal, mask_loss_fn, optimizer, device, training,
                  scaler, ema=None, mask_loss_weight=0.3, id_loss_weight=0.2,
                  accum_steps=1, use_amp=True, epoch=0, num_epochs=50, nt_boost=1.25,
                  metric_names=None, mixup_alpha=0.0, loss_ramp=1.0,
                  global_step=0, warmup_steps=0, base_lr=1e-4):
    model.train(training)
    total_loss = 0.0
    n_finite_steps = 0
    n_skipped = 0
    all_probs, all_labels, all_methods = [], [], []
    all_ious = []
    all_id_preds, all_id_labels = [], []
    n_steps = len(loader)
    if use_amp:
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        amp_ctx = autocast(device_type=device_type, dtype=torch.float16 if device_type == 'cuda' else torch.bfloat16)
    else:
        amp_ctx = nullcontext()
    desc = f"  {'Train' if training else 'Val  '} {epoch + 1:>3d}/{num_epochs}"
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, colour='green' if training else 'cyan')
    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(bar):
        video = batch['video'].to(device, non_blocking=True)
        face_01 = batch['face_01'].to(device, non_blocking=True)
        face_norm = batch['face_norm'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        mask_gt = batch['mask'].to(device, non_blocking=True)
        mask_valid = batch['mask_valid'].to(device, non_blocking=True)
        method_id = batch['method_id'].to(device, non_blocking=True)
        heatmap = batch['landmark_heatmap'].to(device, non_blocking=True)
        methods = batch['method']
        if training and optimizer is not None and warmup_steps > 0:
            if global_step < warmup_steps:
                warm_lr = base_lr * (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = warm_lr
            elif global_step == warmup_steps:
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr
        if training:
            global_step += 1
        with amp_ctx:
            logits, mask_logits, id_logits, details = model(video, face_01, face_norm, landmark_heatmap=heatmap)
            mixed_logits = None
            mix_perm = None
            mix_lam = None
            if training and mixup_alpha > 0.0:
                mix_lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                mix_perm = torch.randperm(labels.shape[0], device=device)
                fused = details['fused']
                fused_mix = mix_lam * fused + (1.0 - mix_lam) * fused[mix_perm]
                mixed_logits = model.classify_embedding(fused_mix)
            sw = None
            if training and nt_boost > 1.0:
                sw = torch.ones(labels.shape[0], dtype=logits.dtype, device=device)
                nt_mask = torch.tensor([m == 'NeuralTextures' for m in methods], device=device, dtype=torch.bool)
                sw = sw.masked_fill(nt_mask, nt_boost)
            loss = focal(logits, labels, sample_weight=sw)
            if mixed_logits is not None:
                mixed_loss = mix_lam * focal(mixed_logits, labels, sample_weight=sw)
                mixed_loss = mixed_loss + (1.0 - mix_lam) * focal(mixed_logits, labels[mix_perm], sample_weight=sw[mix_perm] if sw is not None else None)
                loss = 0.5 * (loss + mixed_loss)
            m_loss = mask_loss_fn(mask_logits, mask_gt, mask_valid)
            id_loss = F.cross_entropy(id_logits, method_id, ignore_index=-1) if id_logits is not None else torch.tensor(0.0, device=device)
            loss = loss + loss_ramp * (mask_loss_weight * m_loss + id_loss_weight * id_loss)
            loss_s = loss / accum_steps

        if not torch.isfinite(loss):
            # Never let a non-finite loss touch the weights, optimizer state, or EMA -
            # GradScaler only guards against inf/nan *gradients*, not a batch whose
            # forward pass already produced a non-finite loss. Skip this batch entirely.
            n_skipped += 1
            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if training and ema is not None:
                healed = _heal_nonfinite_buffers(model, ema.ema)
                if healed:
                    print(f'  [heal] restored {healed} non-finite buffer(s)/param(s) from EMA after a non-finite-loss batch')
            bar.set_postfix(loss='skip-nonfinite', auc=f"{compute_auc(all_labels, all_probs):.3f}" if len(all_labels) > 10 else '—')
            continue

        if training:
            if scaler:
                scaler.scale(loss_s).backward()
            else:
                loss_s.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == n_steps:
                if scaler:
                    scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    # A non-finite grad norm can slip past GradScaler's own inf check
                    # (computed post-unscale, pre-clip) in rare edge cases; treat it the
                    # same way - never apply it, never let it into the EMA.
                    n_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    if scaler:
                        scaler.update()
                    if ema is not None:
                        healed = _heal_nonfinite_buffers(model, ema.ema)
                        if healed:
                            print(f'  [heal] restored {healed} non-finite buffer(s)/param(s) from EMA after a non-finite grad norm')
                    continue
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
        n_finite_steps += 1
        probs = logits.detach().softmax(-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_methods.extend(methods)
        if mask_logits is not None and mask_valid.any():
            iou = compute_mask_iou(mask_logits.detach(), mask_gt, mask_valid)
            if not np.isnan(iou):
                all_ious.append(iou)
        if id_logits is not None:
            valid_id = method_id != -1
            if valid_id.any():
                id_pred = id_logits.detach().argmax(-1)[valid_id].cpu().numpy()
                all_id_preds.extend(id_pred.tolist())
                all_id_labels.extend(method_id[valid_id].cpu().numpy().tolist())
        bar.set_postfix(loss=f'{loss.item():.4f}', auc=f"{compute_auc(all_labels, all_probs):.3f}" if len(all_labels) > 10 else '—')
    bar.close()
    thr = 0.5 if training else select_threshold(np.array(all_probs), np.array(all_labels))
    preds = (np.array(all_probs) >= thr).astype(int)
    acc = float((preds == np.array(all_labels)).mean())
    auc = compute_auc(all_labels, all_probs)
    f1 = float(f1_score(all_labels, preds)) if len(set(all_labels)) > 1 else float('nan')
    pm = per_method_metrics(all_probs, all_labels, all_methods, thr, metric_names=metric_names)
    mask_iou = float(np.mean(all_ious)) if all_ious else float('nan')
    id_acc = float((np.array(all_id_preds) == np.array(all_id_labels)).mean()) if all_id_labels else float('nan')
    id_f1 = float(f1_score(all_id_labels, all_id_preds, average='macro')) if all_id_labels and len(set(all_id_labels)) > 1 else float('nan')
    if n_skipped > 0:
        print(f'  [{"train" if training else "val"}] skipped {n_skipped}/{n_steps} non-finite-loss batches this epoch')
    return {
        'loss': total_loss / max(n_finite_steps, 1), 'acc': acc, 'auc': auc, 'f1': f1,
        'pm': pm, 'thr': thr, 'mask_iou': mask_iou, 'id_acc': id_acc, 'id_f1': id_f1,
        'global_step': global_step, 'n_skipped': n_skipped,
    }


def train_dfb(cfg: dict, train_dl, val_dl, test_dl, model: ForensicMTF):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    save_dir = Path(cfg.get('save_dir', 'records/DFB/models_forensicmtf'))
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    metric_names = cfg.get('metric_names', ['real'] + FF_SUBSETS)
    logger = DFBMetricsLogger(cfg.get('metrics_path', 'records/DFB/results/metrics_forensicmtf.csv'), metric_names=metric_names)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg.get('mem_fraction', 0.95))
        torch.cuda.empty_cache()
    focal = FocalLoss(alpha_real=cfg.get('alpha_real', 0.35), alpha_fake=cfg.get('alpha_fake', 0.65),
                      gamma=cfg.get('focal_gamma', 2.0), label_smoothing=cfg.get('label_smoothing', 0.08))
    mask_loss_fn = MaskLoss()
    ema = ModelEMA(model, decay=cfg.get('ema_decay', 0.999))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 1e-4), weight_decay=cfg.get('weight_decay', 0.03))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    start_ep = 0
    best_auc = 0.0
    best_thr = 0.5
    no_imp = 0
    global_step = 0
    resume = cfg.get('resume', None)
    if resume and Path(resume).exists():
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck['model'])
        ema.ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        if scaler and 'scaler' in ck:
            scaler.load_state_dict(ck['scaler'])
        start_ep = ck.get('epoch', 0) + 1
        best_auc = ck.get('best_auc', 0.0)
        best_thr = ck.get('best_thr', 0.5)
        no_imp = ck.get('no_imp', 0)
        global_step = ck.get('global_step', 0)
    else:
        init_model_path = cfg.get('init_model_path')
        if init_model_path and Path(init_model_path).exists():
            ck = torch.load(init_model_path, map_location=device, weights_only=False)
            state_dict = ck.get('ema', ck.get('model', ck))
            model_state = model.state_dict()
            filtered_state = {}
            skipped_mismatch = []
            for key, value in state_dict.items():
                if key not in model_state:
                    continue
                if model_state[key].shape != value.shape:
                    skipped_mismatch.append(key)
                    continue
                filtered_state[key] = value
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)
            ema.ema.load_state_dict(model.state_dict())
            if missing:
                print(f'Init checkpoint missing keys: {len(missing)}')
            if unexpected:
                print(f'Init checkpoint unexpected keys: {len(unexpected)}')
            if skipped_mismatch:
                print(f'Init checkpoint skipped shape-mismatch keys: {len(skipped_mismatch)}')
            print(f'Initialized training from {init_model_path}')
    n_ep = cfg.get('epochs', 50)
    accum = cfg.get('accum_steps', 4)
    use_amp = cfg.get('use_amp', True)
    mask_loss_weight = cfg.get('mask_loss_weight', 0.3)
    id_loss_weight = cfg.get('id_loss_weight', 0.2)
    warmup_epochs = cfg.get('aux_loss_warmup_epochs', 2)
    nt_boost = cfg.get('nt_boost', 1.25)
    early_stop = cfg.get('early_stop', 8)
    mixup_alpha = cfg.get('fused_mixup_alpha', 0.0)
    base_lr = cfg.get('lr', 1e-4)
    lr_warmup_steps = cfg.get('lr_warmup_steps', 300)
    best_path = save_dir / 'best.pth'
    eb = tqdm(range(start_ep, n_ep), desc='Overall', unit='ep', colour='yellow', dynamic_ncols=True)
    for epoch in eb:
        loss_ramp = min(1.0, (epoch + 1) / max(1, warmup_epochs)) if warmup_epochs > 0 else 1.0
        t0 = time.time()
        train_res = run_epoch_dfb(
            model, train_dl, focal, mask_loss_fn, optimizer, device, True,
            scaler=scaler, ema=ema, mask_loss_weight=mask_loss_weight, id_loss_weight=id_loss_weight,
            accum_steps=accum, use_amp=use_amp, epoch=epoch, num_epochs=n_ep, nt_boost=nt_boost,
            metric_names=metric_names, mixup_alpha=mixup_alpha, loss_ramp=loss_ramp,
            global_step=global_step, warmup_steps=lr_warmup_steps, base_lr=base_lr,
        )
        global_step = train_res['global_step']
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        with torch.no_grad():
            val_res = run_epoch_dfb(
                ema.ema, val_dl, focal, mask_loss_fn, None, device, False, scaler=None, ema=None,
                mask_loss_weight=mask_loss_weight, id_loss_weight=id_loss_weight, accum_steps=1,
                use_amp=use_amp, epoch=epoch, num_epochs=n_ep, metric_names=metric_names, loss_ramp=loss_ramp,
            )
        scheduler.step(val_res['auc'])
        lr = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch + 1,
            'train_loss': f"{train_res['loss']:.5f}", 'train_acc': f"{train_res['acc']:.4f}",
            'train_auc': f"{train_res['auc']:.4f}", 'train_f1': f"{train_res['f1']:.4f}",
            'val_loss': f"{val_res['loss']:.5f}", 'val_acc': f"{val_res['acc']:.4f}",
            'val_auc': f"{val_res['auc']:.4f}", 'val_f1': f"{val_res['f1']:.4f}",
            'val_mask_iou': f"{val_res['mask_iou']:.4f}", 'val_id_acc': f"{val_res['id_acc']:.4f}",
            'val_id_f1': f"{val_res['id_f1']:.4f}", 'lr': f'{lr:.2e}',
        }
        for cls in metric_names:
            row[f'{cls}_acc'] = f"{val_res['pm'].get(cls, {}).get('acc', float('nan')):.4f}"
            row[f'{cls}_auc'] = f"{val_res['pm'].get(cls, {}).get('auc', float('nan')):.4f}"
        logger.write(row)
        val_auc = val_res['auc']
        eb.set_postfix(val=f'{val_auc:.4f}', best=f'{best_auc:.4f}')
        if val_auc > best_auc:
            best_auc = val_auc
            best_thr = val_res['thr']
            no_imp = 0
            torch.save(ema.ema.state_dict(), best_path)
        else:
            no_imp += 1
        ckpt = {
            'epoch': epoch, 'model': model.state_dict(), 'ema': ema.ema.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_auc': best_auc, 'best_thr': best_thr, 'no_imp': no_imp, 'global_step': global_step,
        }
        if scaler:
            ckpt['scaler'] = scaler.state_dict()
        torch.save(ckpt, checkpoints_dir / f'epoch_{epoch + 1:03d}.pth')
        _prune_checkpoints(checkpoints_dir, keep_last=cfg.get('keep_last_checkpoints', 10))
        if early_stop > 0 and no_imp >= early_stop:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        print(f"Epoch {epoch + 1:02d}/{n_ep}: val_auc={val_auc:.4f} val_iou={val_res['mask_iou']:.4f} lr={lr:.1e} [{time.time() - t0:.0f}s]")
    eb.close()
    print('\nTest evaluation on best checkpoint ...')
    ema.ema.load_state_dict(torch.load(best_path, map_location=device))
    with torch.no_grad():
        test_res = run_epoch_dfb(
            ema.ema, test_dl, focal, mask_loss_fn, None, device, False, scaler=None,
            mask_loss_weight=mask_loss_weight, id_loss_weight=id_loss_weight, accum_steps=1,
            use_amp=use_amp, epoch=0, num_epochs=1, metric_names=metric_names,
        )
    print(f"Test AUC={test_res['auc']:.4f} ACC={test_res['acc']:.4f} F1={test_res['f1']:.4f} IoU={test_res['mask_iou']:.4f} thr={best_thr:.4f}")
    for method, values in test_res['pm'].items():
        print(f"  {method:<22}: acc={values['acc']:.4f} auc={values['auc']:.4f}")
    return ema.ema
