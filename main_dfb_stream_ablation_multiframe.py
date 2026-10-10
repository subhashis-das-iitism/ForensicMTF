from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset_multiframe import get_dfb_loaders_multiframe
from forensicmtf.dfb_runner import _eval_model_kwargs, _load_or_build_ffpp_index, find_latest_checkpoint, resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader
from forensicmtf.models.forensicmtf_multiframe import ForensicMTFMultiFrame
from forensicmtf.training.dfb_core import train_dfb

# Re-runs the single-stream rows of main_dfb_stream_ablation.py that were confounded by frame
# count: Frequency-only and Spatial-only both saw one frame there, vs. Temporal's full clip
# (see dfb_dataset_multiframe.py's docstring). Here Stream2/Stream3 get `stream23_frames`
# frames each (mean-pooled before fusion via ForensicMTFMultiFrame), giving a fairer
# single-stream comparison. Temporal-only is NOT re-run -- it already has the full clip and
# is unaffected by this fix; its existing table_stream_ablation.csv numbers still apply.
VARIANTS = {
    'Frequency only (multi-frame)': dict(use_temporal=False, use_noise=True,  use_spatial=False),
    'Spatial only (multi-frame)':   dict(use_temporal=False, use_noise=False, use_spatial=True),
}


def _build_model_mf(train_cfg: dict) -> ForensicMTFMultiFrame:
    return ForensicMTFMultiFrame(
        pretrained=train_cfg.get('pretrained', True), s3_weight=train_cfg.get('s3_weight', 0.3),
        face_size=train_cfg.get('face_size', 112), use_spatial=train_cfg.get('use_spatial', True),
        use_noise=train_cfg.get('use_noise', True), use_temporal=train_cfg.get('use_temporal', True),
        fusion_mode=train_cfg.get('fusion_mode', 'dual_cross_attention'),
        use_localization_head=train_cfg.get('use_localization_head', True),
        use_identification_head=train_cfg.get('use_identification_head', True),
        use_landmark_attention=train_cfg.get('use_landmark_attention', True),
    )


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')


def main():
    parser = argparse.ArgumentParser(description='Fair (multi-frame) re-run of Frequency-only / Spatial-only.')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--compressions', default='c23,c40')
    parser.add_argument('--stream23_frames', type=int, default=8)
    parser.add_argument('--variants', default=None, help=f'Comma-separated subset of {list(VARIANTS)}.')
    args = parser.parse_args()
    compressions = [c.strip() for c in args.compressions.split(',') if c.strip()]
    variants = list(VARIANTS)
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(',')}
        variants = [v for v in VARIANTS if v in wanted]

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    sweep_cfg = cfg.get('ablation', {}).get('sweep', {})
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    out_dir = records_dir / 'stream_ablation_multiframe'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Multi-frame stream ablation: {variants} x {compressions}, stream23_frames={args.stream23_frames}, "
          f"epochs={sweep_cfg.get('epochs', 25)} early_stop={sweep_cfg.get('early_stop', 5)}")

    rows: dict[str, dict[str, dict]] = {name: {} for name in VARIANTS}
    csv_path = out_dir / 'table_stream_ablation_multiframe.csv'
    if csv_path.exists():
        with open(csv_path, newline='') as fh:
            for r in csv.DictReader(fh):
                if r['variant'] in rows:
                    for comp in compressions:
                        if r.get(f'{comp}_auc'):
                            rows[r['variant']].setdefault(comp, {'auc': float(r[f'{comp}_auc']), 'accuracy': float(r[f'{comp}_accuracy'])})

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

            print(f'\n===== {comp}: {name} ({VARIANTS[name]}) =====')
            train_cfg = dict(cfg.get('train', {}))
            train_cfg.update(VARIANTS[name])
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
                'stream23_frames': args.stream23_frames,
            }
            train_dl, val_dl, test_dl = get_dfb_loaders_multiframe(index_by_split, loader_cfg)
            model = _build_model_mf(train_cfg)
            train_dfb(train_cfg, train_dl, val_dl, test_dl, model)

            model_kwargs = _eval_model_kwargs(train_cfg)
            result = evaluate_dfb_loader(best_path, test_dl, out_dir / 'eval' / f'{comp}_{slug}', f'dfb_ffpp_{slug}',
                                          model_kwargs=model_kwargs, model_cls=ForensicMTFMultiFrame)
            m = result['metrics']
            rows[name][comp] = {'auc': m['auc'], 'accuracy': m['accuracy']}
            print(f"  -> {comp}/{name}: AUC={m['auc']:.4f} Acc={m['accuracy']:.4f}")

            with open(csv_path, 'w', newline='') as fh:
                header = ['variant'] + [f'{c}_auc' for c in compressions] + [f'{c}_accuracy' for c in compressions]
                w = csv.writer(fh)
                w.writerow(header)
                for vname in VARIANTS:
                    vals = rows[vname]
                    w.writerow([vname] + [vals.get(c, {}).get('auc', '') for c in compressions]
                               + [vals.get(c, {}).get('accuracy', '') for c in compressions])
            print(f'  Wrote {csv_path}')

    print(f'\nDone. Table: {csv_path}')


if __name__ == '__main__':
    main()
