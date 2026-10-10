from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_eval_loader
from forensicmtf.data.dfb_index import build_dfb_celebdf_index
from forensicmtf.dfb_runner import _eval_model_kwargs, _ffpp_cfg, _index_variant, _load_or_build_ffpp_index, resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader

# s3 is a fixed, non-learned scalar (f_final = f_fused + s3 * f_spatial), so sweeping it
# at eval time on an already-trained checkpoint requires no retraining - just forward
# passes. FF++ is swept on the VAL split (never test) so the chosen value isn't tuned on
# the split we report headline numbers on; CelebDF-v1 has no train/val role at all (the
# model never sees it during training - it's always a zero-shot cross-domain target), so
# sweeping its full test split here is not test-set tuning, just an additional, honestly
# zero-shot data point. Grid brackets both the train-time value (0.3) and the current
# eval-config value (0.7).
S3_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main():
    parser = argparse.ArgumentParser(description='Eval-time sensitivity sweep over the spatial-injection scale s3 (no retraining), on FF++ val and CelebDF-v1 zero-shot.')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--ffpp-split', default='val', choices=['val', 'test'],
                        help='Which FF++ split to sweep on (default: val, to avoid tuning on the reported test split)')
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)

    eval_cfg = dict(cfg.get('evaluate', {}))
    common_cfg = dict(eval_cfg.get('common', {}))
    target_cfg = dict(eval_cfg.get('targets', {}).get('dfb_ffpp', {}))
    merged = {**common_cfg, **target_cfg}
    variant = _index_variant(cfg, smoke=False)

    index_by_split = _load_or_build_ffpp_index(PROJECT_ROOT, cfg, smoke=False)
    ffpp_cfg = _ffpp_cfg(cfg)
    loader_cfg = {
        'n_frames': merged.get('n_frames', ffpp_cfg.get('n_frames', 32)),
        'face_size': merged.get('face_size', ffpp_cfg.get('face_size', 112)),
        'img_size': merged.get('img_size', ffpp_cfg.get('img_size', 224)),
        'batch_size': merged.get('batch_size', 4),
        'num_workers': merged.get('num_workers', 4),
    }
    ffpp_loader = get_dfb_eval_loader(index_by_split[args.ffpp_split], loader_cfg)

    celebdf_root = resolve_path(PROJECT_ROOT, cfg['paths']['celebdf_dfb_root'])
    celebdf_index = build_dfb_celebdf_index(celebdf_root)
    celebdf_loader = get_dfb_eval_loader(celebdf_index['test'], loader_cfg)

    model_path = resolve_path(PROJECT_ROOT, merged['model_path'].format(train_variant=variant))
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    out_root = records_dir / 's3_sweep'
    print(f'Sweeping s3: FF++ {args.ffpp_split} ({len(ffpp_loader.dataset)} samples), CelebDF-v1 test ({len(celebdf_loader.dataset)} samples), checkpoint={model_path}')

    rows = []
    for s3 in S3_GRID:
        model_kwargs = _eval_model_kwargs(merged)
        model_kwargs['s3_weight'] = s3
        ffpp_result = evaluate_dfb_loader(model_path, ffpp_loader, out_root / args.ffpp_split / f's3_{s3:.1f}',
                                          f'dfb_ffpp_{args.ffpp_split}', model_kwargs=model_kwargs)
        celebdf_result = evaluate_dfb_loader(model_path, celebdf_loader, out_root / 'celebdf' / f's3_{s3:.1f}',
                                             'dfb_celebdf', model_kwargs=model_kwargs)
        fm, cm = ffpp_result['metrics'], celebdf_result['metrics']
        row = {'s3': s3, 'ffpp_auc': fm['auc'], 'ffpp_accuracy': fm['accuracy'],
               'celebdf_auc': cm['auc'], 'celebdf_accuracy': cm['accuracy']}
        rows.append(row)
        print(f"  s3={s3:.1f}  FF++ AUC={fm['auc']:.4f} Acc={fm['accuracy']:.4f}  |  CelebDF AUC={cm['auc']:.4f} Acc={cm['accuracy']:.4f}")

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / 'sweep_summary.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['s3', 'ffpp_auc', 'ffpp_accuracy', 'celebdf_auc', 'celebdf_accuracy'])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    best_ffpp = max(rows, key=lambda r: r['ffpp_auc'])
    best_celebdf = max(rows, key=lambda r: r['celebdf_auc'])
    train_value = next(r for r in rows if abs(r['s3'] - 0.3) < 1e-9)
    current_eval_value = next(r for r in rows if abs(r['s3'] - 0.7) < 1e-9)
    print(f'\nWrote {csv_path}')
    print(f"Best FF++ s3={best_ffpp['s3']:.1f} (AUC={best_ffpp['ffpp_auc']:.4f}) | Best CelebDF s3={best_celebdf['s3']:.1f} (AUC={best_celebdf['celebdf_auc']:.4f})")
    print(f"train-time s3=0.3: FF++ AUC={train_value['ffpp_auc']:.4f} CelebDF AUC={train_value['celebdf_auc']:.4f}")
    print(f"current eval-config s3=0.7: FF++ AUC={current_eval_value['ffpp_auc']:.4f} CelebDF AUC={current_eval_value['celebdf_auc']:.4f}")


if __name__ == '__main__':
    main()
