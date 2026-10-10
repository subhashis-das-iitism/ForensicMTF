from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_eval_loader, get_dfb_loaders
from forensicmtf.data.dfb_index import build_dfb_ffpp_index
from forensicmtf.dfb_runner import _build_model, _eval_model_kwargs, _ffpp_cfg, resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader
from forensicmtf.training.dfb_core import train_dfb

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def _filter_by_method(index: list, method: str) -> list:
    return [it for it in index if it['method'] in (method, 'real')]


def main():
    parser = argparse.ArgumentParser(
        description='ForensicMTF cross-manipulation generalization: train on one FF++ manipulation '
                    'method at a time, evaluate against all four (leave-one-out-style, but every method is '
                    'both a training target and a held-out target across the sweep).')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--methods', default=None, help='Comma-separated subset of training methods to run (default: all 4).')
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)

    ffpp_root = resolve_path(PROJECT_ROOT, cfg['paths']['dfb_root']) / 'FaceForensics++'
    ffpp_cfg = _ffpp_cfg(cfg)
    compressions = ffpp_cfg.get('compressions', ['c40'])

    train_methods = METHODS
    if args.methods:
        wanted = {m.strip() for m in args.methods.split(',')}
        train_methods = [m for m in METHODS if m in wanted]
        if not train_methods:
            raise ValueError(f'--methods={args.methods!r} matched none of {METHODS}')

    # Test index covers all 4 methods at once (built once, filtered per training run) - avoids
    # re-walking the dataset tree 4x for what is otherwise the same underlying test split.
    full_test_index = build_dfb_ffpp_index(ffpp_root, 'test', compressions=compressions, fake_subsets=METHODS)

    sweep_cfg = cfg.get('ablation', {}).get('sweep', {})
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    out_dir = records_dir / 'cross_manipulation'
    out_dir.mkdir(parents=True, exist_ok=True)

    loader_cfg_base = {
        'n_frames': ffpp_cfg.get('n_frames', 32),
        'face_size': ffpp_cfg.get('face_size', 112),
        'img_size': ffpp_cfg.get('img_size', 224),
        'batch_size': cfg.get('train', {}).get('batch_size', 4),
        'num_workers': cfg.get('train', {}).get('num_workers', 4),
    }

    print(f"Cross-manipulation sweep: training on {train_methods}, epochs={sweep_cfg.get('epochs', 25)} "
          f"early_stop={sweep_cfg.get('early_stop', 5)}")

    matrix: dict[str, dict[str, float]] = {}
    for train_method in train_methods:
        slug = train_method.lower()
        print(f'\n===== Cross-manipulation: train on {train_method} =====')

        train_index = build_dfb_ffpp_index(ffpp_root, 'train', compressions=compressions, fake_subsets=[train_method])
        val_index = build_dfb_ffpp_index(ffpp_root, 'val', compressions=compressions, fake_subsets=[train_method])
        index_by_split = {
            'train': train_index, 'val': val_index,
            'test': _filter_by_method(full_test_index, train_method),
        }
        n_train_real = sum(1 for it in train_index if it['label'] == 0)
        n_train_fake = sum(1 for it in train_index if it['label'] == 1)
        print(f'  train: real={n_train_real} fake={n_train_fake}')

        train_cfg = dict(cfg.get('train', {}))
        train_cfg['epochs'] = sweep_cfg.get('epochs', 25)
        train_cfg['early_stop'] = sweep_cfg.get('early_stop', 5)
        save_dir = out_dir / f'models_{slug}'
        train_cfg['save_dir'] = str(save_dir)
        train_cfg['metrics_path'] = str(out_dir / 'results' / f'metrics_{slug}.csv')

        loader_cfg = dict(loader_cfg_base, augmentations=train_cfg.get('augmentations'))
        train_dl, val_dl, test_dl = get_dfb_loaders(index_by_split, loader_cfg)
        model = _build_model(train_cfg)
        train_dfb(train_cfg, train_dl, val_dl, test_dl, model)

        model_kwargs = _eval_model_kwargs(train_cfg)
        best_path = save_dir / 'best.pth'

        row: dict[str, float] = {}
        for test_method in METHODS:
            test_items = _filter_by_method(full_test_index, test_method)
            loader = get_dfb_eval_loader(test_items, loader_cfg)
            eval_out = out_dir / 'eval' / f'{slug}_on_{test_method.lower()}'
            result = evaluate_dfb_loader(best_path, loader, eval_out, f'dfb_{test_method.lower()}', model_kwargs=model_kwargs)
            auc = result['metrics']['auc']
            row[test_method] = auc
            tag = '(in-domain)' if test_method == train_method else '(cross-manip)'
            print(f'    -> tested on {test_method:<16} AUC={auc:.4f} {tag}')
        matrix[train_method] = row

    csv_path = out_dir / 'cross_manipulation_matrix.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['train_method'] + METHODS)
        w.writeheader()
        for train_method, row in matrix.items():
            w.writerow({'train_method': train_method, **{m: row.get(m, '') for m in METHODS}})
    print(f'\nWrote {csv_path}')


if __name__ == '__main__':
    main()
