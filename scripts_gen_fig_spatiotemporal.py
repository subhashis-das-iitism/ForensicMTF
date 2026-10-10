"""Generates figures/freq_artefact_comparison.png for the paper from REAL DeepFakeBench
data: one representative face crop per class (real + 4 FF++ C40 manipulation methods),
its FFT log-magnitude spectrum (DC-zeroed), and the four Haar DWT subbands (LL/LH/HL/HH).

Uses the exact same math as forensicmtf.models.detector.FFTBranch._spectrum and
HaarDWTBranch (both are parameter-free / fixed-weight transforms, so no trained
checkpoint is needed - this is not a mockup, it is the literal transform the model
applies to its Stream-2 input).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import load_config
from forensicmtf.data.dfb_index import build_dfb_ffpp_index
from forensicmtf.dfb_runner import resolve_path
from forensicmtf.data.dfb_dataset import _bbox_from_landmarks, _landmark_path_for, _list_frame_files

FACE_SIZE = 112
DC_R = FACE_SIZE // 8

HAAR = torch.tensor([
    [[0.5, 0.5], [0.5, 0.5]],
    [[0.5, 0.5], [-0.5, -0.5]],
    [[0.5, -0.5], [0.5, -0.5]],
    [[0.5, -0.5], [-0.5, 0.5]],
], dtype=torch.float32)  # (4, 2, 2): LL, LH, HL, HH


def load_face_crop(item: dict) -> np.ndarray:
    frame_files = _list_frame_files(item['frames_dir'])
    mid = frame_files[len(frame_files) // 2]
    img = cv2.imread(str(mid))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lm_path = _landmark_path_for(item['landmarks_dir'], mid)
    landmarks = np.load(lm_path).astype(np.float32)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _bbox_from_landmarks(landmarks, w, h)
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
    return crop


def fft_spectrum(face_rgb01: torch.Tensor) -> np.ndarray:
    """face_rgb01: (1,3,H,W) in [0,1]. Reproduces FFTBranch._spectrum exactly."""
    gray = 0.299 * face_rgb01[:, 0] + 0.587 * face_rgb01[:, 1] + 0.114 * face_rgb01[:, 2]
    fft = torch.fft.fftshift(torch.fft.fft2(gray))
    logm = torch.log1p(torch.abs(fft))
    h, w = logm.shape[-2:]
    cy, cx = h // 2, w // 2
    r = DC_R
    mask = torch.ones_like(logm)
    mask[:, max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)] = 0.0
    logm = logm * mask
    mn = logm.flatten(1).min(1).values.view(-1, 1, 1)
    mx = logm.flatten(1).max(1).values.view(-1, 1, 1)
    return ((logm - mn) / (mx - mn + 1e-8))[0].numpy()


def dwt_subbands(face_rgb01: torch.Tensor) -> list[np.ndarray]:
    """face_rgb01: (1,3,H,W). Reproduces HaarDWTBranch's grouped conv, but averaged
    over the 3 RGB channels per subband (for a single-panel grayscale visualization)."""
    haar = HAAR.unsqueeze(1).repeat(3, 1, 1, 1)  # (12,1,2,2), matches HaarDWTBranch
    subbands = F.conv2d(face_rgb01, haar, stride=2, groups=3)  # (1,12,H/2,W/2)
    subbands = subbands.view(1, 3, 4, subbands.shape[-2], subbands.shape[-1])
    out = []
    for k in range(4):  # LL, LH, HL, HH
        band = subbands[0, :, k].abs().mean(0).numpy()
        band = (band - band.min()) / (band.max() - band.min() + 1e-8)
        out.append(band)
    return out


def main():
    cfg = load_config(PROJECT_ROOT / 'config_dfb.yaml')
    ffpp_root = resolve_path(PROJECT_ROOT, cfg['paths']['dfb_root']) / 'FaceForensics++'
    idx = build_dfb_ffpp_index(ffpp_root, 'test', compressions=['c40'],
                               fake_subsets=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'], max_pairs=3)
    by_method: dict[str, dict] = {}
    for it in idx:
        by_method.setdefault(it['method'], it)
    row_order = ['real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    labels = {'real': 'Real', 'Deepfakes': 'Deepfakes', 'Face2Face': 'Face2Face',
              'FaceSwap': 'FaceSwap', 'NeuralTextures': 'NeuralTextures'}

    col_titles = ['Face crop', 'FFT log-mag.\n(DC-zeroed)', 'LL', 'LH', 'HL', 'HH']
    fig, axes = plt.subplots(len(row_order), 6, figsize=(11, 9.5))

    for r, method in enumerate(row_order):
        item = by_method[method]
        crop = load_face_crop(item)
        face01 = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        spectrum = fft_spectrum(face01)
        subbands = dwt_subbands(face01)

        axes[r, 0].imshow(crop)
        axes[r, 1].imshow(spectrum, cmap='inferno')
        for k in range(4):
            axes[r, 2 + k].imshow(subbands[k], cmap='gray')

        for c in range(6):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(labels[method], fontsize=11, rotation=90)

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10)

    fig.tight_layout()
    out_path = PROJECT_ROOT / 'records' / 'DFB' / 'figures' / 'freq_artefact_comparison.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
