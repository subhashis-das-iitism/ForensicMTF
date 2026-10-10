from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import FACE_NORM, VIDEO_NORM, get_dfb_eval_loader
from forensicmtf.data.dfb_index import build_dfb_ffpp_index, load_index, save_index
from forensicmtf.dfb_runner import (
    _eval_model_kwargs,
    _ffpp_cfg,
    _index_dir,
    expand_config_value,
    resolve_path,
)
from forensicmtf.evaluation.generic import compute_eer, youden_threshold
from forensicmtf.models.forensicmtf import ForensicMTF


def _load_or_build_ffpp_index_for_compression(project_root: Path, cfg: dict, compression: str) -> dict:
    """Loads (or builds) the FF++ index for a SPECIFIC compression, independent of whatever
    cfg['dfb']['ffpp']['compressions'] currently says - dfb_runner._load_or_build_ffpp_index
    always uses the config's (single, fixed) compressions list, which silently gives you the
    wrong split (e.g. C40 frames) if you ask to evaluate a C23 checkpoint without also editing
    the config. This resolves the C23-checkpoint-evaluated-on-C40-data bug that produced a
    spuriously low 'clean' AUC (0.8516) on the first robustness_c23.csv run."""
    variant = f'ffpp_{compression}'
    path = _index_dir(project_root, cfg) / f'{variant}.json'
    if path.exists():
        return load_index(path)
    ffpp_root = resolve_path(project_root, cfg['paths']['dfb_root']) / 'FaceForensics++'
    ffpp_fake_subsets = _ffpp_cfg(cfg).get('fake_subsets', ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'])
    index_by_split = {}
    for split in ('train', 'val', 'test'):
        index_by_split[split] = build_dfb_ffpp_index(ffpp_root, split, compressions=[compression], fake_subsets=ffpp_fake_subsets)
    save_index(index_by_split, path)
    return index_by_split

# Reused directly from forensicmtf.data.dfb_dataset's VIDEO_NORM/FACE_NORM constants,
# reshaped for broadcasting against (B,C,T,H,W) / (B,C,H,W) tensors respectively.
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


def _gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


def _gaussian_blur(x: torch.Tensor, kernel: int, sigma: float) -> torch.Tensor:
    if x.dim() == 5:  # (B, C, T, H, W) video clip - blur each frame independently
        b, c, t, h, w = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = TF.gaussian_blur(flat, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
        return flat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return TF.gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def _jpeg_compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Real JPEG encode/decode round-trip (via PIL), not a simulated approximation -
    matches FFDBFGC's Table IV protocol (quality factor QF), applied per-frame/per-face."""
    import io
    from PIL import Image
    orig_shape = x.shape
    h, w = orig_shape[-2:]
    flat = x.reshape(-1, 3, h, w)
    out = torch.empty_like(flat)
    for i in range(flat.shape[0]):
        arr = (flat[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype('uint8')
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        dec = np.array(Image.open(buf).convert('RGB')).astype('float32') / 255.0
        out[i] = torch.from_numpy(dec).permute(2, 0, 1).to(x.device)
    return out.reshape(orig_shape)


def _salt_pepper(x: torch.Tensor, density: float, mode: str = 'both') -> torch.Tensor:
    # A single per-pixel (channel-shared) mask, so a "salt"/"pepper" site is a true
    # white/black dot rather than independent per-channel noise. mode='both' splits the
    # density evenly between salt and pepper (as before); mode='salt'/'pepper' applies
    # only that polarity at the full requested density, for isolating which one the
    # model is more sensitive to.
    if x.dim() == 5:
        b, c, t, h, w = x.shape
        r = torch.rand(b, 1, t, h, w, device=x.device)
    else:
        b, c, h, w = x.shape
        r = torch.rand(b, 1, h, w, device=x.device)
    out = x.clone()
    if mode == 'both':
        salt = (r < density / 2).expand_as(x)
        pepper = ((r >= density / 2) & (r < density)).expand_as(x)
    elif mode == 'salt':
        salt = (r < density).expand_as(x)
        pepper = torch.zeros_like(salt)
    elif mode == 'pepper':
        salt = torch.zeros_like(x, dtype=torch.bool)
        pepper = (r < density).expand_as(x)
    else:
        raise ValueError(mode)
    out = torch.where(salt, torch.ones_like(out), out)
    out = torch.where(pepper, torch.zeros_like(out), out)
    return out


def _median_filter(x: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Per-channel 2D median filter (the classical, textbook defense against impulse/
    salt-and-pepper noise specifically - unlike a mean/Gaussian filter, the median is not
    dragged toward the extreme value an impulse pixel takes). Implemented via unfold for
    GPU throughput rather than a per-frame OpenCV/scipy call."""
    orig_shape = x.shape
    h, w = orig_shape[-2:]
    flat = x.reshape(-1, 1, h, w)
    pad = kernel // 2
    padded = F.pad(flat, [pad, pad, pad, pad], mode='reflect')
    patches = padded.unfold(2, kernel, 1).unfold(3, kernel, 1)
    patches = patches.contiguous().view(*patches.shape[:4], -1)
    median = patches.median(dim=-1).values
    return median.reshape(orig_shape)


PERTURBATIONS = {
    'clean': None,
    'gaussian_noise': lambda x: _gaussian_noise(x, sigma=0.10),
    'gaussian_blur': lambda x: _gaussian_blur(x, kernel=7, sigma=2.0),
    'salt_pepper': lambda x: _salt_pepper(x, density=0.05, mode='both'),
    'salt_only': lambda x: _salt_pepper(x, density=0.05, mode='salt'),
    'pepper_only': lambda x: _salt_pepper(x, density=0.05, mode='pepper'),
    'jpeg_qf50': lambda x: _jpeg_compress(x, quality=50),
    # Median-filter mitigation variants: apply the same impulse perturbation, then a 3x3
    # median filter, before the model sees it - isolates how much of the salt-and-pepper
    # vulnerability (Table XII) a trivial classical defense recovers.
    'clean_medfilt3': lambda x: _median_filter(x, kernel=3),
    'salt_pepper_medfilt3': lambda x: _median_filter(_salt_pepper(x, density=0.05, mode='both'), kernel=3),
    'salt_only_medfilt3': lambda x: _median_filter(_salt_pepper(x, density=0.05, mode='salt'), kernel=3),
    'pepper_only_medfilt3': lambda x: _median_filter(_salt_pepper(x, density=0.05, mode='pepper'), kernel=3),
}


def _apply(name: str, video: torch.Tensor, face_01: torch.Tensor):
    fn = PERTURBATIONS[name]
    if fn is None:
        return video, face_01, _renorm_face(face_01)
    video01 = _denorm_video(video)
    video01_p = fn(video01)
    face01_p = fn(face_01)
    return _renorm_video(video01_p), face01_p, _renorm_face(face01_p)


@torch.no_grad()
def _run_one(model, loader, device, name: str):
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc=f'Robustness[{name}]', leave=False):
        video = batch['video'].to(device)
        face_01 = batch['face_01'].to(device)
        heatmap = batch['landmark_heatmap'].to(device)
        labels = batch['label']

        video_p, face_01_p, face_norm_p = _apply(name, video, face_01)
        logits, _, _, _ = model(video_p, face_01_p, face_norm_p, landmark_heatmap=heatmap)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    y = np.array(all_labels)
    p = np.nan_to_num(np.array(all_probs, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    from sklearn.metrics import roc_curve
    auc = roc_auc_score(y, p)
    fpr, tpr, thr_arr = roc_curve(y, p)
    eer = compute_eer(fpr, tpr)
    thr = youden_threshold(fpr, tpr, thr_arr)
    preds = (p >= thr).astype(int)
    acc = accuracy_score(y, preds)
    return {'perturbation': name, 'auc': round(float(auc), 4), 'eer': round(float(eer), 4),
            'accuracy': round(float(acc), 4), 'threshold': round(float(thr), 4), 'n': len(y)}


def main():
    parser = argparse.ArgumentParser(description='ForensicMTF robustness-under-perturbation eval (FF++ test split).')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--compressions', default='c40', help="Which trained checkpoint to evaluate ('c40' or 'c23').")
    args = parser.parse_args()

    torch.manual_seed(42)  # fixed seed for reproducible noise/salt-pepper draws across reruns
    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_cfg = dict(cfg.get('evaluate', {}))
    common_cfg = dict(eval_cfg.get('common', {}))
    target_cfg = dict(eval_cfg.get('targets', {}).get('dfb_ffpp', {}))
    merged = {**common_cfg, **target_cfg}

    train_variant = f'ffpp_{args.compressions}'
    index_by_split = _load_or_build_ffpp_index_for_compression(PROJECT_ROOT, cfg, args.compressions)
    split = merged.get('split', 'test')
    ffpp_cfg = _ffpp_cfg(cfg)
    loader_cfg = {
        'n_frames': merged.get('n_frames', ffpp_cfg.get('n_frames', 32)),
        'face_size': merged.get('face_size', ffpp_cfg.get('face_size', 112)),
        'img_size': merged.get('img_size', ffpp_cfg.get('img_size', 224)),
        'batch_size': merged.get('batch_size', 4),
        'num_workers': merged.get('num_workers', 4),
    }
    loader = get_dfb_eval_loader(index_by_split[split], loader_cfg)

    model_path = resolve_path(PROJECT_ROOT, expand_config_value(merged['model_path'], train_variant=train_variant))
    model = ForensicMTF(pretrained=False, **_eval_model_kwargs(merged)).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) and ('ema' in ckpt or 'model' in ckpt) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    rows = []
    for name in PERTURBATIONS:
        row = _run_one(model, loader, device, name)
        rows.append(row)
        print(f"  {name:<16}: AUC={row['auc']:.4f} EER={row['eer']:.4f} Acc={row['accuracy']:.4f}")

    clean_auc = rows[0]['auc']
    for row in rows:
        row['delta_auc'] = round(clean_auc - row['auc'], 4)

    out_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB')) / 'robustness'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'robustness_{args.compressions}.csv'
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['perturbation', 'auc', 'delta_auc', 'eer', 'accuracy', 'threshold', 'n'])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
