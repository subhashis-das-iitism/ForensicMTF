"""Robustness performance-curve sweep (AUC vs. perturbation severity), in the visual
style of MVIM's Fig. 7 and FFDBFGC's Table IV: for each perturbation type, evaluate
across several severity levels and plot AUC/Accuracy vs. severity. Unlike MVIM's Fig. 7
(which overlays 7 competing methods' curves, since they had access to those baselines'
outputs), this plots ONLY ForensicMTF's own curve - we do not have FFD/D&L/M2TR/
SFFs/SDIML/LVNet's trained weights or code, so a multi-method overlay is not possible
here; see the paper text for this caveat.

Severity axes:
  - Gaussian blur: kernel size {3,7,11,15,19,23} (odd, unambiguous parameter - directly
    comparable in spirit to MVIM's {5,11,17,23,29}, though sigma is fixed at 2.0 here
    rather than swept jointly).
  - Gaussian noise: sigma in OUR [0,1]-pixel-scale convention {0.0,0.02,0.05,0.08,0.10,0.15}
    - NOT claimed to be on the same numerical scale as MVIM's sigma (their pixel-value
      convention, 0-255 vs [0,1], is not stated precisely enough in the source to match
      exactly).
  - JPEG quality factor: {100,85,70,55,40}, matching FFDBFGC's Table IV exactly (an
    unambiguous, standard parameter).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_eval_loader
from forensicmtf.dfb_runner import _eval_model_kwargs, _ffpp_cfg, resolve_path
from forensicmtf.models.forensicmtf import ForensicMTF

from main_dfb_robustness import (
    _denorm_video,
    _gaussian_blur,
    _gaussian_noise,
    _jpeg_compress,
    _load_or_build_ffpp_index_for_compression,
    _median_filter,
    _renorm_face,
    _renorm_video,
)

SWEEPS = {
    'gaussian_blur': {'param': 'kernel', 'values': [3, 7, 11, 15, 19, 23],
                      'fn': lambda x, v: _gaussian_blur(x, kernel=v, sigma=2.0)},
    'gaussian_noise': {'param': 'sigma', 'values': [0.0, 0.02, 0.05, 0.08, 0.10, 0.15],
                       'fn': lambda x, v: _gaussian_noise(x, sigma=v) if v > 0 else x},
    'jpeg_compression': {'param': 'quality_factor', 'values': [100, 85, 70, 55, 40],
                         'fn': lambda x, v: _jpeg_compress(x, quality=v) if v < 100 else x},
}


def _apply(fn, video: torch.Tensor, face_01: torch.Tensor, value, medfilt: bool = False):
    video01 = _denorm_video(video)
    video01_p = fn(video01, value)
    face01_p = fn(face_01, value)
    if medfilt:
        # Median-filter mitigation: same 3x3 classical defense main_dfb_robustness.py applies
        # to salt-and-pepper, now applied as a post-perturbation preprocessing step before the
        # model sees gaussian_blur/gaussian_noise/jpeg_compression inputs.
        video01_p = _median_filter(video01_p, kernel=3)
        face01_p = _median_filter(face01_p, kernel=3)
    return _renorm_video(video01_p), face01_p, _renorm_face(face01_p)


@torch.no_grad()
def _run_one(model, loader, device, fn, value, desc: str, medfilt: bool = False):
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        video = batch['video'].to(device)
        face_01 = batch['face_01'].to(device)
        heatmap = batch['landmark_heatmap'].to(device)
        labels = batch['label']
        video_p, face_01_p, face_norm_p = _apply(fn, video, face_01, value, medfilt=medfilt)
        logits, _, _, _ = model(video_p, face_01_p, face_norm_p, landmark_heatmap=heatmap)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    y = np.array(all_labels)
    p = np.nan_to_num(np.array(all_probs, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    auc = roc_auc_score(y, p)
    preds = (p >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    return {'auc': round(float(auc), 4), 'accuracy': round(float(acc), 4)}


def main():
    parser = argparse.ArgumentParser(description='ForensicMTF robustness severity-sweep + curve plot (FF++ C40 test split).')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--compressions', default='c40')
    parser.add_argument('--medfilt', action='store_true',
                        help='Apply a 3x3 median-filter mitigation step after each perturbation, before the model sees the input.')
    args = parser.parse_args()
    tag = 'medfilt3' if args.medfilt else None

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_cfg = dict(cfg.get('evaluate', {}))
    common_cfg = dict(eval_cfg.get('common', {}))
    target_cfg = dict(eval_cfg.get('targets', {}).get('dfb_ffpp', {}))
    merged = {**common_cfg, **target_cfg}

    train_variant = f'ffpp_{args.compressions}'
    # NOTE: must use the compression-specific index loader, not dfb_runner._load_or_build_ffpp_index
    # (which always builds/loads the config's fixed cfg['dfb']['ffpp']['compressions'] split regardless
    # of --compressions). See main_dfb_robustness.py's _load_or_build_ffpp_index_for_compression
    # docstring -- using the wrong one here silently evaluates the C23 checkpoint on C40 frames,
    # producing a spuriously low clean AUC (~0.85 instead of ~0.997). Confirmed reproduced on this file
    # before this fix (2026-08-10).
    index_by_split = _load_or_build_ffpp_index_for_compression(PROJECT_ROOT, cfg, args.compressions)
    ffpp_cfg = _ffpp_cfg(cfg)
    loader_cfg = {
        'n_frames': merged.get('n_frames', ffpp_cfg.get('n_frames', 32)),
        'face_size': merged.get('face_size', ffpp_cfg.get('face_size', 112)),
        'img_size': merged.get('img_size', ffpp_cfg.get('img_size', 224)),
        'batch_size': merged.get('batch_size', 4),
        'num_workers': merged.get('num_workers', 4),
    }
    loader = get_dfb_eval_loader(index_by_split[merged.get('split', 'test')], loader_cfg)

    model_path = resolve_path(PROJECT_ROOT, f'records/DFB/models_{train_variant}/best.pth')
    model = ForensicMTF(pretrained=False, **_eval_model_kwargs(merged)).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) and ('ema' in ckpt or 'model' in ckpt) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    out_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB')) / 'robustness'
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f'_{tag}' if tag else ''
    results = {}
    for name, spec in SWEEPS.items():
        rows = []
        for value in spec['values']:
            res = _run_one(model, loader, device, spec['fn'], value, desc=f'{name}={value}{suffix}', medfilt=args.medfilt)
            rows.append({spec['param']: value, 'auc': res['auc'], 'accuracy': res['accuracy']})
            print(f'  {name}{suffix} {spec["param"]}={value}: AUC={res["auc"]:.4f} Acc={res["accuracy"]:.4f}')
        results[name] = rows
        with open(out_dir / f'curve_{name}{suffix}_{args.compressions}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=[spec['param'], 'auc', 'accuracy'])
            w.writeheader()
            for row in rows:
                w.writerow(row)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (name, spec) in enumerate(SWEEPS.items()):
        rows = results[name]
        xs = [r[spec['param']] for r in rows]
        aucs = [r['auc'] for r in rows]
        accs = [r['accuracy'] for r in rows]
        axes[0, col].plot(xs, aucs, marker='o', color='#e74c3c', label=f'ForensicMTF{" +MedFilt3" if args.medfilt else ""}')
        axes[0, col].set_title(f'{name.replace("_", " ").title()}')
        axes[0, col].set_xlabel(spec['param'])
        axes[0, col].set_ylabel('AUC')
        axes[0, col].set_ylim(0.3, 1.0)
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(alpha=0.3)
        if name == 'jpeg_compression':
            axes[0, col].invert_xaxis()

        axes[1, col].plot(xs, accs, marker='o', color='#e74c3c', label=f'ForensicMTF{" +MedFilt3" if args.medfilt else ""}')
        axes[1, col].set_xlabel(spec['param'])
        axes[1, col].set_ylabel('Accuracy')
        axes[1, col].set_ylim(0.3, 1.0)
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(alpha=0.3)
        if name == 'jpeg_compression':
            axes[1, col].invert_xaxis()

    fig.tight_layout()
    fig_path = out_dir / f'robustness_curves{suffix}_{args.compressions}.png'
    fig.savefig(fig_path, dpi=200)
    print(f'\nWrote {fig_path}')


if __name__ == '__main__':
    main()
