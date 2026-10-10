from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from tqdm.auto import tqdm

from forensicmtf.evaluation.generic import _plot_confusion, _plot_roc, _plot_score_dist, compute_eer, youden_threshold
from forensicmtf.models.forensicmtf import ForensicMTF
from forensicmtf.training.core import per_method_metrics
from forensicmtf.training.dfb_core import compute_mask_f1_per_sample, compute_mask_iou_per_sample


def _plot_localization_examples(examples: list, out_path: Path, max_examples: int = 8) -> None:
    """examples: list of dicts with keys face/gt/pred/method/iou. One example per
    manipulation method (see collection logic in evaluate_dfb_loader below) rather than
    just the first N valid samples encountered, so the figure represents every method
    -- including weaker ones -- instead of whichever samples happened to load first."""
    n = min(len(examples), max_examples)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        ex = examples[i]
        face, gt, pred = ex['face'], ex['gt'], ex['pred']
        method, iou = ex.get('method', ''), ex.get('iou')
        axes[i, 0].imshow(face)
        axes[i, 0].set_title(method or 'face'); axes[i, 0].axis('off')
        axes[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('GT mask'); axes[i, 1].axis('off')
        axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Pred mask (IoU={iou:.2f})' if iou is not None else 'Pred mask'); axes[i, 2].axis('off')
        # Overlay: predicted mask as a semi-transparent red region on top of the face crop,
        # usually more legible than a separate black/white panel for judging how the
        # predicted region relates to actual facial features.
        axes[i, 3].imshow(face)
        overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = pred * 0.45
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('overlay'); axes[i, 3].axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_dfb_loader(model_path, loader, out_dir, dataset_name: str, model_kwargs: dict | None = None,
                         block_streams: set | None = None, model_cls=ForensicMTF):
    """model_cls: model class to instantiate (default ForensicMTF). Pass
    ForensicMTFMultiFrame (forensicmtf_multiframe.py) when model_path was trained with
    a multi-frame face_01/face_norm stack -- the base class's extract_features doesn't reshape
    the (B,K,3,H,W) stack and will crash on it."""
    """block_streams: optional subset of {'temporal', 'frequency', 'spatial'} -- when given,
    that stream's raw input is zeroed before the forward pass, so an already-trained
    (unmodified) checkpoint can be probed for how much it relies on each stream, without
    retraining. 'temporal' zeroes the video clip (Stream1), 'frequency' zeroes the face crop
    fed to the noise/frequency stream (Stream2: BayarConv2d/FFT/Haar-DWT), 'spatial' zeroes
    both the normalized face crop and the landmark heatmap fed to Stream3. This always keeps
    every stream module constructed and every tensor non-None, so fusion still takes its
    normally-trained code path (e.g. dual_cross_attention's f_v/f_n branch) rather than
    falling back to an untrained fusion submodule -- see main_dfb_stream_block_eval.py."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    block_streams = block_streams or set()
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault('pretrained', False)
    model = model_cls(**model_kwargs).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) and ('ema' in ckpt or 'model' in ckpt) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_probs, all_labels, all_paths = [], [], []
    ious = []
    ious_by_method: dict[str, list] = {}
    mask_f1s = []
    mask_f1s_by_method: dict[str, list] = {}
    id_preds, id_labels = [], []
    localization_examples_by_method: dict[str, dict] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Eval {dataset_name}', leave=False):
            video = batch['video'].to(device)
            face_01 = batch['face_01'].to(device)
            face_norm = batch['face_norm'].to(device)
            heatmap = batch['landmark_heatmap'].to(device)
            mask_gt = batch['mask'].to(device)
            mask_valid = batch['mask_valid'].to(device)
            method_id = batch['method_id'].to(device)

            if 'temporal' in block_streams:
                video = torch.zeros_like(video)
            if 'frequency' in block_streams:
                face_01 = torch.zeros_like(face_01)
            if 'spatial' in block_streams:
                face_norm = torch.zeros_like(face_norm)
                heatmap = torch.zeros_like(heatmap)

            logits, mask_logits, id_logits, _ = model(video, face_01, face_norm, landmark_heatmap=heatmap)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            methods = batch.get('method', [f'{dataset_name}_{i}' for i in range(len(probs))])
            all_probs.extend(probs.tolist())
            all_labels.extend(batch['label'].numpy().tolist())
            all_paths.extend(methods)

            if mask_logits is not None and mask_valid.any():
                per_sample_iou = compute_mask_iou_per_sample(mask_logits, mask_gt, mask_valid)
                per_sample_f1 = compute_mask_f1_per_sample(mask_logits, mask_gt, mask_valid)
                if per_sample_iou is not None:
                    valid_idx_list = mask_valid.nonzero(as_tuple=True)[0].tolist()
                    logits_up = F.interpolate(mask_logits, size=mask_gt.shape[-2:], mode='bilinear', align_corners=False)
                    pred_mask_full = (torch.sigmoid(logits_up) > 0.5).float()
                    for j, sample_idx in enumerate(valid_idx_list):
                        iou_val = float(per_sample_iou[j].item())
                        ious.append(iou_val)
                        method_name = methods[sample_idx]
                        ious_by_method.setdefault(method_name, []).append(iou_val)
                        f1_val = float(per_sample_f1[j].item())
                        mask_f1s.append(f1_val)
                        mask_f1s_by_method.setdefault(method_name, []).append(f1_val)
                        # Qualitative figure: keep exactly one example per manipulation
                        # method (the first one encountered in eval order for that method,
                        # not cherry-picked for IoU quality), so the figure represents every
                        # method -- including our weaker categories -- rather than whichever
                        # samples happened to load first.
                        if method_name not in localization_examples_by_method:
                            # face_01 is normally (B,3,H,W); the multi-frame stream ablation
                            # (dfb_dataset_multiframe.py) instead carries (B,K,3,H,W) -- display
                            # the middle frame, matching the single-frame mask/spatial_map convention.
                            face_vis = face_01[sample_idx]
                            if face_vis.dim() == 4:
                                face_vis = face_vis[face_vis.shape[0] // 2]
                            localization_examples_by_method[method_name] = {
                                'face': face_vis.permute(1, 2, 0).clamp(0, 1).cpu().numpy(),
                                'gt': mask_gt[sample_idx, 0].cpu().numpy(),
                                'pred': pred_mask_full[sample_idx, 0].cpu().numpy(),
                                'method': method_name,
                                'iou': iou_val,
                            }

            if id_logits is not None:
                valid_id = method_id != -1
                if valid_id.any():
                    id_pred = id_logits.argmax(-1)[valid_id].cpu().numpy()
                    id_preds.extend(id_pred.tolist())
                    id_labels.extend(method_id[valid_id].cpu().numpy().tolist())

    y = np.array(all_labels)
    p_raw = np.array(all_probs)
    n_nonfinite = int((~np.isfinite(p_raw)).sum())
    if n_nonfinite:
        print(f'WARNING: {n_nonfinite}/{len(p_raw)} predicted probabilities were non-finite (NaN/inf) - '
              f'a bug upstream (likely in the checkpoint being evaluated) produced degenerate outputs on some '
              f'inputs. Substituting 0.5 for scoring so this reports rather than crashes.')
    p = np.nan_to_num(p_raw.astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    fpr, tpr, thr_arr = roc_curve(y, p)
    eer = compute_eer(fpr, tpr)
    best_thr = youden_threshold(fpr, tpr, thr_arr)
    preds = (p >= best_thr).astype(int)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds) if len(np.unique(y)) > 1 else float('nan')
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    summary = {
        'dataset': dataset_name,
        'auc': round(float(auc), 4),
        'ap': round(float(ap), 4),
        'eer': round(float(eer), 4),
        'accuracy': round(float(acc), 4),
        'f1': round(float(f1), 4) if f1 == f1 else f1,
        'real_acc': round(float(real_acc), 4),
        'fake_acc': round(float(fake_acc), 4),
        'threshold': round(float(best_thr), 4),
        'n_real': int((y == 0).sum()),
        'n_fake': int((y == 1).sum()),
    }
    # Only report mask_iou / id_acc / id_f1 when this eval set actually has the ground
    # truth to make them meaningful - e.g. Celeb-DF-v1 has no forgery masks at all, and
    # its fake samples have no FF++ method label, so id_labels only ever contains the
    # 'real' class there and an identification accuracy/F1 against a single true class
    # isn't a meaningful signal. Omit the columns entirely rather than report nan.
    if ious:
        summary['mask_iou'] = round(float(np.mean(ious)), 4)
        summary['mask_f1'] = round(float(np.mean(mask_f1s)), 4)
    if id_labels and len(set(id_labels)) > 1:
        id_acc = float((np.array(id_preds) == np.array(id_labels)).mean())
        id_f1 = float(f1_score(id_labels, id_preds, average='macro'))
        summary['id_acc'] = round(id_acc, 4)
        summary['id_f1'] = round(id_f1, 4)
    with open(out / 'metrics_summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # Per-method breakdown (classification acc/auc via the shared per_method_metrics
    # helper, plus per-method mask IoU) - method names are read directly from the data
    # rather than assumed to be the FF++ 4-method set, so this works unmodified for any
    # dataset (FaceShifter/DeepFakeDetection/UADFV/DFDC/Celeb-DF all use different
    # method label conventions).
    unique_methods = sorted(set(all_paths))
    metric_names = (['real'] if 'real' in unique_methods else []) + [m for m in unique_methods if m != 'real']
    pm = per_method_metrics(all_probs, all_labels, all_paths, best_thr, metric_names=metric_names) if metric_names else {}
    per_method_rows = []
    for method in metric_names:
        row = {'method': method, 'n': int(sum(1 for m in all_paths if m == method))}
        if method in pm:
            row['acc'] = round(pm[method]['acc'], 4)
            row['auc'] = round(pm[method]['auc'], 4)
        method_ious = ious_by_method.get(method)
        if method_ious:
            row['mask_iou'] = round(float(np.mean(method_ious)), 4)
            row['mask_iou_n'] = len(method_ious)
            method_f1s = mask_f1s_by_method.get(method, [])
            row['mask_f1'] = round(float(np.mean(method_f1s)), 4)
        per_method_rows.append(row)
    if per_method_rows:
        fieldnames = sorted({k for row in per_method_rows for k in row.keys()}, key=lambda k: (k != 'method', k != 'n', k))
        with open(out / 'per_method_metrics.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, restval='')
            w.writeheader()
            for row in per_method_rows:
                w.writerow(row)

    with open(out / 'per_video_scores.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method', 'label', 'prob', 'pred', 'correct'])
        w.writeheader()
        for method, lbl, prob in zip(all_paths, all_labels, all_probs):
            pred = int(prob >= best_thr)
            w.writerow({'method': method, 'label': int(lbl), 'prob': round(float(prob), 6), 'pred': pred, 'correct': int(pred == int(lbl))})

    pretty = dataset_name.upper()
    _plot_roc(fpr, tpr, auc, out / 'roc_curve.png', f'{pretty} - ROC Curve')
    _plot_score_dist(p, y, best_thr, out / 'score_distribution.png', f'{pretty} - Score Distribution')
    _plot_confusion(cm, out / 'confusion_matrix.png', f'{pretty} - Confusion Matrix')
    if localization_examples_by_method:
        # Fixed canonical order when present (matches the method order used elsewhere in
        # this paper's tables); any other method name sorts alphabetically after those.
        _canonical_order = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        ordered_methods = sorted(
            localization_examples_by_method,
            key=lambda m: (_canonical_order.index(m) if m in _canonical_order else len(_canonical_order), m),
        )
        localization_examples = [localization_examples_by_method[m] for m in ordered_methods]
        _plot_localization_examples(localization_examples, out / 'localization_examples.png')
    return {'metrics': summary, 'per_method': per_method_rows, 'out_dir': str(out)}
