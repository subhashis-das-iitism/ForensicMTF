from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_loaders
from forensicmtf.dfb_runner import _build_model, _eval_model_kwargs, _ffpp_cfg, _load_or_build_ffpp_index, resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader
from forensicmtf.training.dfb_core import train_dfb

# 'dual_cross_attention' is excluded from the *default* sweep: it is our headline fusion
# mode, already trained as the main FF++(C40) checkpoint under the main train.epochs=50/
# early_stop=8 budget, not this ablation's epochs=25/early_stop=5 budget -- so its existing
# numbers are NOT a matched comparison against the other three modes here. Pass
# `--modes dual_cross_attention` (or include it in a custom --modes list) to explicitly run
# it under the same 25-epoch/early_stop=5 protocol as the rest of this table.
FUSION_MODES = ['addition', 'concatenation', 'attention']
ALL_FUSION_MODES = FUSION_MODES + ['dual_cross_attention']


def main():
    parser = argparse.ArgumentParser(
        description='ForensicMTF fusion-strategy ablation: retrain with each alternative '
                    'fusion_mode (addition / concatenation / self-attention) on FF++(C40), holding '
                    'everything else fixed, to compare against the headline dual_cross_attention model.')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--modes', default=None,
                         help='Comma-separated subset of ALL_FUSION_MODES to run (including, optionally, '
                              "'dual_cross_attention', which the default sweep excludes).")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)

    modes = FUSION_MODES
    if args.modes:
        wanted = {m.strip() for m in args.modes.split(',')}
        modes = [m for m in ALL_FUSION_MODES if m in wanted]
        if not modes:
            raise ValueError(f'--modes={args.modes!r} matched none of {ALL_FUSION_MODES}')

    index_by_split = _load_or_build_ffpp_index(PROJECT_ROOT, cfg, smoke=False)
    ffpp_cfg = _ffpp_cfg(cfg)
    sweep_cfg = cfg.get('ablation', {}).get('sweep', {})
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    out_dir = records_dir / 'fusion_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    loader_cfg_base = {
        'n_frames': ffpp_cfg.get('n_frames', 32),
        'face_size': ffpp_cfg.get('face_size', 112),
        'img_size': ffpp_cfg.get('img_size', 224),
        'batch_size': cfg.get('train', {}).get('batch_size', 4),
        'num_workers': cfg.get('train', {}).get('num_workers', 4),
    }

    print(f"Fusion-strategy sweep: {modes}, epochs={sweep_cfg.get('epochs', 25)} early_stop={sweep_cfg.get('early_stop', 5)}")

    rows = []
    for mode in modes:
        print(f'\n===== Fusion mode: {mode} =====')
        train_cfg = dict(cfg.get('train', {}))
        train_cfg['fusion_mode'] = mode
        train_cfg['epochs'] = sweep_cfg.get('epochs', 25)
        train_cfg['early_stop'] = sweep_cfg.get('early_stop', 5)
        save_dir = out_dir / f'models_{mode}'
        train_cfg['save_dir'] = str(save_dir)
        train_cfg['metrics_path'] = str(out_dir / 'results' / f'metrics_{mode}.csv')

        loader_cfg = dict(loader_cfg_base, augmentations=train_cfg.get('augmentations'))
        train_dl, val_dl, test_dl = get_dfb_loaders(index_by_split, loader_cfg)
        model = _build_model(train_cfg)
        train_dfb(train_cfg, train_dl, val_dl, test_dl, model)

        model_kwargs = _eval_model_kwargs(train_cfg)
        best_path = save_dir / 'best.pth'
        result = evaluate_dfb_loader(best_path, test_dl, out_dir / 'eval' / f'dfb_ffpp_{mode}', 'dfb_ffpp', model_kwargs=model_kwargs)
        m = result['metrics']
        row = {'fusion_mode': mode, 'auc': m['auc'], 'accuracy': m['accuracy'], 'f1': m.get('f1', float('nan'))}
        rows.append(row)
        print(f"  -> {mode}: AUC={m['auc']:.4f} Acc={m['accuracy']:.4f}")

    # Merge into any existing table rather than overwrite -- running a --modes subset (e.g. just
    # 'dual_cross_attention') must not clobber the other, already-verified rows in this file.
    csv_path = out_dir / 'fusion_ablation_table.csv'
    existing = {}
    if csv_path.exists():
        with open(csv_path, newline='') as fh:
            for r in csv.DictReader(fh):
                existing[r['fusion_mode']] = r
    for row in rows:
        existing[row['fusion_mode']] = row
    # Keep the original FUSION_MODES order first, then any extra modes (e.g. dual_cross_attention)
    # appended in the order they were run.
    ordered_names = [m for m in FUSION_MODES if m in existing]
    ordered_names += [m for m in existing if m not in ordered_names]
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['fusion_mode', 'auc', 'accuracy', 'f1'])
        w.writeheader()
        for name in ordered_names:
            w.writerow(existing[name])
    print(f'\nWrote {csv_path}')


if __name__ == '__main__':
    main()
