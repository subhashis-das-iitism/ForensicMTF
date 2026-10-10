from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from forensicmtf.data.dfb_dataset import get_dfb_eval_loader, get_dfb_loaders
from forensicmtf.data.dfb_index import build_dfb_celebdf_index, build_dfb_ffpp_index, load_index, save_index
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader
from forensicmtf.models.forensicmtf import ForensicMTF
from forensicmtf.training.dfb_core import train_dfb


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (CPU+CUDA) for a reproducible run -- used only when an
    explicit --seed is passed (e.g. the confidence-interval reruns), so the default,
    already-verified headline runs stay exactly as nondeterministic/unseeded as before."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(save_dir: Path) -> Path | None:
    """Highest-numbered *loadable* checkpoints/epoch_NNN.pth under save_dir, or None if none
    exist. Used to auto-resume an ablation combo that got killed mid-training (e.g. by an
    environment restart) instead of silently restarting it from epoch 0 -- train_dfb's
    --resume already restores full optimizer/scheduler/scaler/epoch state from one of these,
    this just finds which one to pass. Skips over a checkpoint that fails to load: if the
    process was killed mid-write to the newest epoch_NNN.pth, torch.save's write isn't atomic,
    leaving a truncated/corrupt (sometimes literally 0-byte) file -- falls back to the next
    most recent one instead of crashing."""
    ckpt_dir = Path(save_dir) / 'checkpoints'
    if not ckpt_dir.is_dir():
        return None
    for ckpt in sorted(ckpt_dir.glob('epoch_*.pth'), reverse=True):
        try:
            torch.load(ckpt, map_location='cpu', weights_only=False)
        except Exception as exc:
            print(f'  [skip] {ckpt} failed to load ({exc}) -- likely truncated by an interrupted save; trying an earlier checkpoint')
            continue
        return ckpt
    return None


def resolve_path(project_root: Path, value: str | None):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def expand_config_value(value: str | None, **kwargs):
    if value is None:
        return None
    return value.format(**kwargs)


def _ffpp_cfg(cfg: dict) -> dict:
    return cfg.get('dfb', {}).get('ffpp', {})


def _index_dir(project_root: Path, cfg: dict) -> Path:
    records_dir = resolve_path(project_root, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    d = records_dir / 'index'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_variant(cfg: dict, smoke: bool) -> str:
    if smoke:
        return 'smoke'
    ffpp_cfg = _ffpp_cfg(cfg)
    comps = '_'.join(ffpp_cfg.get('compressions', ['c40']))
    return f'ffpp_{comps}'


def build_dfb_ffpp_index_by_split(project_root: Path, cfg: dict, smoke: bool = False) -> dict:
    ffpp_root = resolve_path(project_root, cfg['paths']['dfb_root']) / 'FaceForensics++'
    ffpp_cfg = _ffpp_cfg(cfg)
    compressions = ffpp_cfg.get('compressions', ['c40'])
    fake_subsets = ffpp_cfg.get('fake_subsets', ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'])
    smoke_cfg = cfg.get('dfb', {}).get('smoke', {})
    max_pairs = {
        'train': smoke_cfg.get('train_pairs', 20),
        'val': smoke_cfg.get('val_pairs', 5),
        'test': smoke_cfg.get('test_pairs', 5),
    }
    index_by_split = {}
    for split in ('train', 'val', 'test'):
        index_by_split[split] = build_dfb_ffpp_index(
            ffpp_root, split, compressions=compressions, fake_subsets=fake_subsets,
            max_pairs=max_pairs[split] if smoke else None,
        )
    out_path = _index_dir(project_root, cfg) / f'{_index_variant(cfg, smoke)}.json'
    save_index(index_by_split, out_path)
    for split, items in index_by_split.items():
        n_real = sum(1 for it in items if it['label'] == 0)
        n_fake = sum(1 for it in items if it['label'] == 1)
        print(f'[{split}] real={n_real} fake={n_fake} total={len(items)}  -> {out_path}')
    return index_by_split


def _load_or_build_ffpp_index(project_root: Path, cfg: dict, smoke: bool) -> dict:
    path = _index_dir(project_root, cfg) / f'{_index_variant(cfg, smoke)}.json'
    if path.exists():
        return load_index(path)
    return build_dfb_ffpp_index_by_split(project_root, cfg, smoke=smoke)


def _build_model(train_cfg: dict) -> ForensicMTF:
    return ForensicMTF(
        pretrained=train_cfg.get('pretrained', True),
        s3_weight=train_cfg.get('s3_weight', 0.3),
        face_size=train_cfg.get('face_size', 112),
        use_spatial=train_cfg.get('use_spatial', True),
        use_noise=train_cfg.get('use_noise', True),
        use_temporal=train_cfg.get('use_temporal', True),
        fusion_mode=train_cfg.get('fusion_mode', 'dual_cross_attention'),
        denoise_noise_stream=train_cfg.get('denoise_noise_stream', False),
        denoise_kernel=train_cfg.get('denoise_kernel', 3),
        denoise_sigma=train_cfg.get('denoise_sigma', 0.6),
        feature_dropout_prob=train_cfg.get('feature_dropout_prob', 0.0),
        use_noise_attention=train_cfg.get('use_noise_attention', False),
        classifier_hidden_dims=train_cfg.get('classifier_hidden_dims'),
        use_localization_head=train_cfg.get('use_localization_head', True),
        use_identification_head=train_cfg.get('use_identification_head', True),
        use_landmark_attention=train_cfg.get('use_landmark_attention', True),
    )


def _slugify(name: str) -> str:
    slug = name.lower()
    for ch in ' ()/,+':
        slug = slug.replace(ch, '_' if ch != '+' else 'plus_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug.strip('_')


def _eval_model_kwargs(merged: dict) -> dict:
    return dict(
        s3_weight=merged.get('s3_weight', 0.7),
        face_size=merged.get('face_size', 112),
        use_spatial=merged.get('use_spatial', True),
        use_noise=merged.get('use_noise', True),
        use_temporal=merged.get('use_temporal', True),
        fusion_mode=merged.get('fusion_mode', 'dual_cross_attention'),
        use_localization_head=merged.get('use_localization_head', True),
        use_identification_head=merged.get('use_identification_head', True),
        use_landmark_attention=merged.get('use_landmark_attention', True),
    )


def run_dfb_training(project_root: Path, cfg: dict, smoke: bool = False, resume: str | None = None,
                      seed: int | None = None):
    train_cfg = dict(cfg.get('train', {}))
    variant = _index_variant(cfg, smoke)
    format_kwargs = {'train_variant': variant}
    save_dir = expand_config_value(train_cfg['save_dir'], **format_kwargs)
    metrics_path = expand_config_value(train_cfg['metrics_path'], **format_kwargs)
    if seed is not None:
        # Tagged, separate output so a confidence-interval reseed run never overwrites the
        # existing verified (unseeded) headline checkpoint/metrics for the same variant.
        save_dir = f'{save_dir}_seed{seed}'
        stem, _, ext = metrics_path.rpartition('.')
        metrics_path = f'{stem}_seed{seed}.{ext}' if stem else f'{metrics_path}_seed{seed}'
        set_seed(seed)
    train_cfg['save_dir'] = str(resolve_path(project_root, save_dir))
    train_cfg['metrics_path'] = str(resolve_path(project_root, metrics_path))
    if resume:
        train_cfg['resume'] = resume
    if smoke:
        smoke_cfg = cfg.get('dfb', {}).get('smoke', {})
        train_cfg['epochs'] = smoke_cfg.get('epochs', 2)
        train_cfg['batch_size'] = smoke_cfg.get('batch_size', 2)
        train_cfg['accum_steps'] = smoke_cfg.get('accum_steps', 2)
        train_cfg['n_frames'] = smoke_cfg.get('n_frames', 16)
        train_cfg['early_stop'] = 0

    index_by_split = _load_or_build_ffpp_index(project_root, cfg, smoke)
    ffpp_cfg = _ffpp_cfg(cfg)
    loader_cfg = {
        'n_frames': train_cfg.get('n_frames', ffpp_cfg.get('n_frames', 32)),
        'face_size': train_cfg.get('face_size', ffpp_cfg.get('face_size', 112)),
        'img_size': train_cfg.get('img_size', ffpp_cfg.get('img_size', 224)),
        'batch_size': train_cfg.get('batch_size', 4),
        'num_workers': train_cfg.get('num_workers', 4),
        'augmentations': train_cfg.get('augmentations'),
    }
    train_dl, val_dl, test_dl = get_dfb_loaders(index_by_split, loader_cfg)
    model = _build_model(train_cfg)
    return train_dfb(train_cfg, train_dl, val_dl, test_dl, model)


def run_dfb_evaluation(project_root: Path, cfg: dict, smoke: bool = False, seed: int | None = None):
    eval_cfg = dict(cfg.get('evaluate', {}))
    common_cfg = dict(eval_cfg.get('common', {}))
    target_cfg = dict(eval_cfg.get('targets', {}).get('dfb_ffpp', {}))
    merged = {**common_cfg, **target_cfg}
    variant = _index_variant(cfg, smoke)
    if seed is not None:
        # Match the '_seed<N>'-suffixed checkpoint/output paths run_dfb_training used for this seed.
        variant = f'{variant}_seed{seed}'

    index_by_split = _load_or_build_ffpp_index(project_root, cfg, smoke)
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
    format_kwargs = {'train_variant': variant, 'dataset': 'dfb_ffpp'}
    model_path = resolve_path(project_root, expand_config_value(merged['model_path'], **format_kwargs))
    out_dir = resolve_path(project_root, expand_config_value(merged['out_dir'], **format_kwargs))
    return evaluate_dfb_loader(model_path, loader, out_dir, 'dfb_ffpp', model_kwargs=_eval_model_kwargs(merged))


def run_dfb_cross_domain_evaluation(project_root: Path, cfg: dict, smoke: bool = False):
    eval_cfg = dict(cfg.get('evaluate', {}))
    common_cfg = dict(eval_cfg.get('common', {}))
    target_cfg = dict(eval_cfg.get('targets', {}).get('dfb_celebdf_cross_domain', {}))
    merged = {**common_cfg, **target_cfg}
    variant = _index_variant(cfg, smoke)

    celebdf_root = resolve_path(project_root, cfg['paths']['celebdf_dfb_root'])
    index = build_dfb_celebdf_index(celebdf_root)
    loader_cfg = {
        'n_frames': merged.get('n_frames', 32),
        'face_size': merged.get('face_size', 112),
        'img_size': merged.get('img_size', 224),
        'batch_size': merged.get('batch_size', 4),
        'num_workers': merged.get('num_workers', 4),
    }
    loader = get_dfb_eval_loader(index['test'], loader_cfg)
    format_kwargs = {'train_variant': variant, 'dataset': 'dfb_celebdf'}
    model_path = resolve_path(project_root, expand_config_value(merged['model_path'], **format_kwargs))
    out_dir = resolve_path(project_root, expand_config_value(merged['out_dir'], **format_kwargs))
    return evaluate_dfb_loader(model_path, loader, out_dir, 'dfb_celebdf', model_kwargs=_eval_model_kwargs(merged))


def run_dfb_ablation_variant(project_root: Path, cfg: dict, variant: dict, index_by_split: dict,
                             celebdf_index: dict, sweep_cfg: dict) -> dict:
    """Train + evaluate one MLG-MTF ablation variant (FF++ test split + Celeb-DF-v1
    cross-domain), writing outputs under records/DFB/ablation/. Returns a flat dict of
    metrics (ffpp_* and celebdf_* prefixed) for aggregation into the sweep table."""
    name = variant['name']
    slug = variant.get('dataset_variant') or _slugify(name)

    train_cfg = dict(cfg.get('train', {}))
    train_cfg.update(variant.get('train', {}))
    train_cfg['epochs'] = sweep_cfg.get('epochs', 25)
    train_cfg['early_stop'] = sweep_cfg.get('early_stop', 5)

    records_dir = resolve_path(project_root, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    ablation_dir = records_dir / 'ablation'
    save_dir = ablation_dir / f'models_{slug}'
    train_cfg['save_dir'] = str(save_dir)
    train_cfg['metrics_path'] = str(ablation_dir / 'results' / f'metrics_{slug}.csv')

    ffpp_cfg = _ffpp_cfg(cfg)
    loader_cfg = {
        'n_frames': train_cfg.get('n_frames', ffpp_cfg.get('n_frames', 32)),
        'face_size': train_cfg.get('face_size', ffpp_cfg.get('face_size', 112)),
        'img_size': train_cfg.get('img_size', ffpp_cfg.get('img_size', 224)),
        'batch_size': train_cfg.get('batch_size', 4),
        'num_workers': train_cfg.get('num_workers', 4),
        'augmentations': train_cfg.get('augmentations'),
    }
    print(f'\n===== Ablation variant: {name} ({slug}) =====')
    print(f"  localization={train_cfg.get('use_localization_head', True)} "
          f"identification={train_cfg.get('use_identification_head', True)} "
          f"landmark_attn={train_cfg.get('use_landmark_attention', True)} "
          f"epochs={train_cfg['epochs']} early_stop={train_cfg['early_stop']}")

    train_dl, val_dl, test_dl = get_dfb_loaders(index_by_split, loader_cfg)
    model = _build_model(train_cfg)
    train_dfb(train_cfg, train_dl, val_dl, test_dl, model)

    model_kwargs = _eval_model_kwargs(train_cfg)
    best_path = save_dir / 'best.pth'

    ffpp_out = ablation_dir / 'eval' / f'dfb_ffpp_{slug}'
    ffpp_result = evaluate_dfb_loader(best_path, test_dl, ffpp_out, 'dfb_ffpp', model_kwargs=model_kwargs)

    celebdf_loader = get_dfb_eval_loader(celebdf_index['test'], loader_cfg)
    celebdf_out = ablation_dir / 'eval' / f'dfb_celebdf_{slug}'
    celebdf_result = evaluate_dfb_loader(best_path, celebdf_loader, celebdf_out, 'dfb_celebdf', model_kwargs=model_kwargs)

    row = {'name': name, 'variant': slug}
    for key, value in ffpp_result['metrics'].items():
        if key != 'dataset':
            row[f'ffpp_{key}'] = value
    for key, value in celebdf_result['metrics'].items():
        if key != 'dataset':
            row[f'celebdf_{key}'] = value
    return row


def run_dfb_flat_eval(project_root: Path, cfg: dict, items: list, dataset_name: str, train_variant: str) -> dict:
    """Evaluate an already-trained checkpoint (records/DFB/models_<train_variant>/best.pth)
    against an arbitrary flat item list (no train/val split needed - this is always a
    zero-shot cross-domain/cross-method eval, never used for training). Used for
    additional DeepFakeBench targets beyond the primary FF++/Celeb-DF-v1 pair, e.g.
    FaceShifter, DeepFakeDetection, UADFV - see main_dfb_cross_eval.py."""
    ffpp_cfg = _ffpp_cfg(cfg)
    eval_common = dict(cfg.get('evaluate', {}).get('common', {}))
    loader_cfg = {
        'n_frames': eval_common.get('n_frames', ffpp_cfg.get('n_frames', 32)),
        'face_size': eval_common.get('face_size', ffpp_cfg.get('face_size', 112)),
        'img_size': eval_common.get('img_size', ffpp_cfg.get('img_size', 224)),
        'batch_size': eval_common.get('batch_size', 4),
        'num_workers': eval_common.get('num_workers', 4),
    }
    loader = get_dfb_eval_loader(items, loader_cfg)
    records_dir = resolve_path(project_root, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    model_path = records_dir / f'models_{train_variant}' / 'best.pth'
    out_dir = records_dir / 'eval' / f'{dataset_name}_{train_variant}'
    model_kwargs = _eval_model_kwargs(eval_common)
    return evaluate_dfb_loader(model_path, loader, out_dir, dataset_name, model_kwargs=model_kwargs)
