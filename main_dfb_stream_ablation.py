from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_loaders
from forensicmtf.dfb_runner import _build_model, _eval_model_kwargs, _load_or_build_ffpp_index, find_latest_checkpoint, resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader
from forensicmtf.training.dfb_core import train_dfb

# Proper (from-scratch) T/S/N stream-contribution ablation -- unlike main_dfb_stream_block_eval.py
# (which blocks streams at inference time on the already-trained headline checkpoint), each
# variant here is a genuinely separate training run built with only the listed stream(s)
# constructed (via ForensicMTF's use_temporal/use_noise/use_spatial flags), so the
# fusion/classifier weights are learned specifically for that stream combination rather than
# repurposed from a model trained expecting all three. Naming matches the codebase's own
# T(emporal)/N(oise, i.e. frequency: BayarConv2d+FFT+Haar-DWT)/S(patial) terminology.
# 'Full model' is intentionally excluded: it's the existing headline checkpoint
# (records/DFB/models_ffpp_<c23,c40>/best.pth, epochs=50/early_stop=8) -- reused directly
# rather than retrained under this ablation's shorter epochs=25/early_stop=5 sweep budget,
# same convention as main_dfb_fusion_ablation.py's dual_cross_attention exclusion.
STREAM_COMBOS = {
    'Temporal only':  dict(use_temporal=True,  use_noise=False, use_spatial=False),
    'Frequency only': dict(use_temporal=False, use_noise=True,  use_spatial=False),
    'Spatial only':   dict(use_temporal=False, use_noise=False, use_spatial=True),
    'T+F (no S)':     dict(use_temporal=True,  use_noise=True,  use_spatial=False),
    'T+S (no F)':     dict(use_temporal=True,  use_noise=False, use_spatial=True),
    'F+S (no T)':     dict(use_temporal=False, use_noise=True,  use_spatial=True),
}


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')


def main():
    parser = argparse.ArgumentParser(
        description='Proper (from-scratch) T/S/N stream-contribution ablation on FF++ HQ (c23) and LQ (c40).')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--compressions', default='c23,c40', help="Comma-separated compressions to run, e.g. 'c23,c40'.")
    parser.add_argument('--variants', default=None, help=f'Comma-separated subset of {list(STREAM_COMBOS)} to run.')
    args = parser.parse_args()
    compressions = [c.strip() for c in args.compressions.split(',') if c.strip()]

    variants = list(STREAM_COMBOS)
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(',')}
        variants = [v for v in STREAM_COMBOS if v in wanted]
        if not variants:
            raise ValueError(f'--variants={args.variants!r} matched none of {list(STREAM_COMBOS)}')

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    sweep_cfg = cfg.get('ablation', {}).get('sweep', {})
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    out_dir = records_dir / 'stream_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stream ablation: {variants} x {compressions}, epochs={sweep_cfg.get('epochs', 25)} early_stop={sweep_cfg.get('early_stop', 5)}")

    # rows[variant][compression] = {'auc':..., 'accuracy':...}
    rows: dict[str, dict[str, dict]] = {name: {} for name in STREAM_COMBOS}
    csv_path = out_dir / 'table_stream_ablation.csv'
    if csv_path.exists():
        with open(csv_path, newline='') as fh:
            for r in csv.DictReader(fh):
                name = r['variant']
                if name not in rows:
                    continue
                for comp in compressions:
                    if r.get(f'{comp}_auc'):
                        rows[name].setdefault(comp, {'auc': float(r[f'{comp}_auc']), 'accuracy': float(r[f'{comp}_accuracy'])})

    for comp in compressions:
        run_cfg = dict(cfg)
        run_cfg['dfb'] = dict(cfg.get('dfb', {}))
        run_cfg['dfb']['ffpp'] = dict(cfg['dfb']['ffpp'])
        run_cfg['dfb']['ffpp']['compressions'] = [comp]
        index_by_split = _load_or_build_ffpp_index(PROJECT_ROOT, run_cfg, smoke=False)

        ffpp_cfg = run_cfg['dfb']['ffpp']
        for name in variants:
            slug = _slug(name)
            save_dir = out_dir / f'models_{comp}_{slug}'
            best_path = save_dir / 'best.pth'
            metrics_summary = out_dir / 'eval' / f'{comp}_{slug}' / 'metrics_summary.csv'
            if metrics_summary.exists():
                print(f'[skip] {comp} / {name}: already have {metrics_summary}')
                continue

            print(f'\n===== {comp}: {name} ({STREAM_COMBOS[name]}) =====')
            train_cfg = dict(cfg.get('train', {}))
            train_cfg.update(STREAM_COMBOS[name])
            train_cfg['epochs'] = sweep_cfg.get('epochs', 25)
            train_cfg['early_stop'] = sweep_cfg.get('early_stop', 5)
            train_cfg['save_dir'] = str(save_dir)
            train_cfg['metrics_path'] = str(out_dir / 'results' / f'metrics_{comp}_{slug}.csv')

            latest_ckpt = find_latest_checkpoint(save_dir)
            if latest_ckpt is not None:
                train_cfg['resume'] = str(latest_ckpt)
                print(f'  Resuming from {latest_ckpt}')

            loader_cfg = {
                'n_frames': ffpp_cfg.get('n_frames', 32), 'face_size': ffpp_cfg.get('face_size', 112),
                'img_size': ffpp_cfg.get('img_size', 224), 'batch_size': train_cfg.get('batch_size', 4),
                'num_workers': train_cfg.get('num_workers', 4), 'augmentations': train_cfg.get('augmentations'),
            }
            train_dl, val_dl, test_dl = get_dfb_loaders(index_by_split, loader_cfg)
            model = _build_model(train_cfg)
            train_dfb(train_cfg, train_dl, val_dl, test_dl, model)

            model_kwargs = _eval_model_kwargs(train_cfg)
            result = evaluate_dfb_loader(best_path, test_dl, out_dir / 'eval' / f'{comp}_{slug}', f'dfb_ffpp_{slug}',
                                          model_kwargs=model_kwargs)
            m = result['metrics']
            rows[name][comp] = {'auc': m['auc'], 'accuracy': m['accuracy']}
            print(f"  -> {comp}/{name}: AUC={m['auc']:.4f} Acc={m['accuracy']:.4f}")

            # Write the table after every run so partial progress is never lost on a crash.
            with open(csv_path, 'w', newline='') as fh:
                header = ['variant'] + [f'{c}_auc' for c in compressions] + [f'{c}_accuracy' for c in compressions]
                w = csv.writer(fh)
                w.writerow(header)
                for vname in STREAM_COMBOS:
                    vals = rows[vname]
                    w.writerow([vname] + [vals.get(c, {}).get('auc', '') for c in compressions]
                               + [vals.get(c, {}).get('accuracy', '') for c in compressions])
            print(f'  Wrote {csv_path}')

    print(f'\nDone. Table: {csv_path}')


if __name__ == '__main__':
    main()
