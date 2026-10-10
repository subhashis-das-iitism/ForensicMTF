"""Frequency-band masking ablation, in the spirit of CUTA's Table IX/X (Shi et al. 2025,
"Customized Transformer Adapter With Frequency Masking for Deepfake Detection"): at TEST
TIME ONLY (no retraining), randomly zero X% of the 2D FFT coefficients within a chosen
frequency band of the input, then evaluate zero-shot cross-domain AUC, to see how much the
already-trained model's cross-domain performance depends on which frequency content is
intact.

NOTE ON METHODOLOGY: CUTA's own band boundaries are defined as literal quadrants of the
*unshifted* 2D FFT grid (Low/Mid/High along increasing row/col index). Their PDF's text
layer was partially garbled at the exact boundary formulas, so we do not claim pixel-exact
reproduction of their band definitions. We instead define bands by radial distance from
the DC component after fftshift (the more common, resolution-independent convention):
Low = innermost 25% of the radius, Mid = 25-75%, High = outer 25%, All = every coefficient
outside DC. This is an analogous, not identical, ablation - reported as such in the paper.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import FACE_NORM, VIDEO_NORM, get_dfb_eval_loader
from forensicmtf.data.dfb_index import build_dfb_celebdf_index, build_dfb_dfdc_index
from forensicmtf.dfb_runner import _eval_model_kwargs, resolve_path
from forensicmtf.models.forensicmtf import ForensicMTF

_VIDEO_MEAN = torch.tensor(VIDEO_NORM.mean).view(1, 3, 1, 1, 1)
_VIDEO_STD = torch.tensor(VIDEO_NORM.std).view(1, 3, 1, 1, 1)
_FACE_MEAN = torch.tensor(FACE_NORM.mean).view(1, 3, 1, 1)
_FACE_STD = torch.tensor(FACE_NORM.std).view(1, 3, 1, 1)


def _denorm_video(video: torch.Tensor) -> torch.Tensor:
    return (video * _VIDEO_STD.to(video.device) + _VIDEO_MEAN.to(video.device)).clamp(0.0, 1.0)


def _renorm_video(video01: torch.Tensor) -> torch.Tensor:
    return (video01 - _VIDEO_MEAN.to(video01.device)) / _VIDEO_STD.to(video01.device)


def _renorm_face(face01: torch.Tensor) -> torch.Tensor:
    return (face01 - _FACE_MEAN.to(face01.device)) / _FACE_STD.to(face01.device)


def _band_mask(h: int, w: int, band: str, device) -> torch.Tensor:
    """Boolean (h,w) mask, True where a coefficient belongs to `band` (post-fftshift,
    DC at center). Uses Chebyshev (L-infinity) distance from center, normalized so the
    band reaches 1.0 along the full edge midpoint in every direction (not only at the
    four corners, unlike an earlier Euclidean/L2 version whose 'high' band selected only
    tiny near-empty corner slivers and masked to a no-op equal to the true baseline).

    Band boundaries are EQUAL-AREA, not equal-width-in-radius: since a Chebyshev ball's
    area grows as r^2, naively equal-width radius bins (0-0.25 / 0.25-0.75 / 0.75-1.0)
    give area fractions 6% / 50% / 44% - low and high end up wildly different sizes,
    so 'mask 15% of the band' would zero a very different absolute number of
    coefficients depending on which band, confounding the very comparison this ablation
    is trying to make. Solving r1^2 = 1/3, r2^2 = 2/3 for the boundaries instead gives
    low/mid/high exactly 1/3 of the area each, so a fixed within-band masking ratio
    corresponds to the same number of removed coefficients in every band."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    cy, cx = h / 2.0, w / 2.0
    r = torch.maximum((yy - cy).abs() / cy, (xx - cx).abs() / cx)
    r1 = (1.0 / 3.0) ** 0.5   # ~0.577, area fraction 1/3
    r2 = (2.0 / 3.0) ** 0.5   # ~0.816, cumulative area fraction 2/3
    if band == 'low':
        return r < r1
    if band == 'mid':
        return (r >= r1) & (r < r2)
    if band == 'high':
        return r >= r2
    if band == 'all':
        return r >= 0.0
    raise ValueError(band)


def _freq_mask_image(x01: torch.Tensor, band: str, ratio: float) -> torch.Tensor:
    """x01: (..., 3, H, W) in [0,1]. Randomly zeroes `ratio` fraction of the FFT
    coefficients within `band` (independently per sample/frame/channel), then inverse-FFTs
    back to a real image. ratio=0 is a no-op (returns x01 unchanged, up to FFT round-trip
    floating-point noise)."""
    if ratio <= 0.0:
        return x01
    orig_shape = x01.shape
    h, w = orig_shape[-2:]
    flat = x01.reshape(-1, h, w)
    fft = torch.fft.fftshift(torch.fft.fft2(flat), dim=(-2, -1))

    band_mask = _band_mask(h, w, band, x01.device)  # (h, w), True = in-band
    rand = torch.rand(flat.shape[0], h, w, device=x01.device)
    drop = band_mask.unsqueeze(0) & (rand < ratio)  # per-sample random subset of the band

    fft = torch.where(drop, torch.zeros_like(fft), fft)
    recon = torch.fft.ifft2(torch.fft.ifftshift(fft, dim=(-2, -1))).real
    return recon.reshape(orig_shape).clamp(0.0, 1.0)


def _apply(band: str, ratio: float, video: torch.Tensor, face_01: torch.Tensor):
    if ratio <= 0.0:
        return video, face_01, _renorm_face(face_01)
    video01 = _denorm_video(video)
    video01_p = _freq_mask_image(video01, band, ratio)
    face01_p = _freq_mask_image(face_01, band, ratio)
    return _renorm_video(video01_p), face01_p, _renorm_face(face01_p)


@torch.no_grad()
def _run_one(model, loader, device, band: str, ratio: float, desc: str):
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        video = batch['video'].to(device)
        face_01 = batch['face_01'].to(device)
        heatmap = batch['landmark_heatmap'].to(device)
        labels = batch['label']

        video_p, face_01_p, face_norm_p = _apply(band, ratio, video, face_01)
        logits, _, _, _ = model(video_p, face_01_p, face_norm_p, landmark_heatmap=heatmap)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    y = np.array(all_labels)
    p = np.nan_to_num(np.array(all_probs, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    auc = roc_auc_score(y, p) if len(set(y.tolist())) > 1 else float('nan')
    preds = (p >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    return {'auc': round(float(auc), 4), 'accuracy': round(float(acc), 4), 'n': len(y)}


def _load_model(model_path: Path, model_kwargs: dict, device) -> ForensicMTF:
    model = ForensicMTF(pretrained=False, **model_kwargs).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) and ('ema' in ckpt or 'model' in ckpt) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Frequency-band masking cross-domain ablation (CUTA-inspired), C40-trained checkpoint.')
    parser.add_argument('--config', default='config_dfb.yaml')
    args = parser.parse_args()

    torch.manual_seed(42)  # fixed seed for reproducible per-pixel masking draws across reruns
    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_cfg = dict(cfg.get('evaluate', {}))
    common_cfg = dict(eval_cfg.get('common', {}))
    merged = dict(common_cfg)
    loader_cfg = {
        'n_frames': merged.get('n_frames', 32), 'face_size': merged.get('face_size', 112),
        'img_size': merged.get('img_size', 224), 'batch_size': merged.get('batch_size', 4),
        'num_workers': merged.get('num_workers', 4),
    }

    model_path = resolve_path(PROJECT_ROOT, 'records/DFB/models_ffpp_c40/best.pth')
    model = _load_model(model_path, _eval_model_kwargs(merged), device)

    celebdf_v1_root = resolve_path(PROJECT_ROOT, cfg['paths']['celebdf_dfb_root'])
    celebdf_v2_root = resolve_path(PROJECT_ROOT, cfg['paths']['celebdf_v2_dfb_root'])
    dfdc_root = resolve_path(PROJECT_ROOT, cfg['paths']['dfdc_root'])
    targets = {
        'Celeb-DF-v1': get_dfb_eval_loader(build_dfb_celebdf_index(celebdf_v1_root)['test'], loader_cfg),
        'Celeb-DF-v2': get_dfb_eval_loader(build_dfb_celebdf_index(celebdf_v2_root)['test'], loader_cfg),
        'DFDC': get_dfb_eval_loader(build_dfb_dfdc_index(dfdc_root), loader_cfg),
    }

    out_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB')) / 'freqmask_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Table A: fixed ratio=15%, vary band (matches CUTA's Table IX structure)
    rows_band = []
    for band in ['low', 'mid', 'high', 'all']:
        row = {'band': band, 'ratio': 0.15}
        for target_name, loader in targets.items():
            res = _run_one(model, loader, device, band, 0.15, desc=f'FreqMask[{band}@15%] {target_name}')
            row[target_name] = res['auc']
            print(f'  band={band:<4} ratio=15% {target_name:<12}: AUC={res["auc"]:.4f}')
        rows_band.append(row)
    with open(out_dir / 'freqmask_by_band.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['band', 'ratio'] + list(targets.keys()))
        w.writeheader()
        for row in rows_band:
            w.writerow(row)

    # Table B: fixed band='high' (CUTA's own chosen default), vary ratio
    rows_ratio = []
    for ratio in [0.0, 0.15, 0.30, 0.50, 0.70]:
        row = {'band': 'high', 'ratio': ratio}
        for target_name, loader in targets.items():
            res = _run_one(model, loader, device, 'high', ratio, desc=f'FreqMask[high@{int(ratio*100)}%] {target_name}')
            row[target_name] = res['auc']
            print(f'  band=high  ratio={ratio:.0%} {target_name:<12}: AUC={res["auc"]:.4f}')
        rows_ratio.append(row)
    with open(out_dir / 'freqmask_by_ratio.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['band', 'ratio'] + list(targets.keys()))
        w.writeheader()
        for row in rows_ratio:
            w.writerow(row)

    print(f'\nWrote {out_dir / "freqmask_by_band.csv"} and {out_dir / "freqmask_by_ratio.csv"}')


if __name__ == '__main__':
    main()
