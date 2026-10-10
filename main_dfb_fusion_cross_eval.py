from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_eval_loader
from forensicmtf.dfb_runner import resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader
from main_dfb_cross_eval import TARGETS, _build_items  # noqa: E402  (reuse the existing target-index builders)

# Cross-domain generalization comparison across all four fusion strategies from the
# fusion-ablation sweep (main_dfb_fusion_ablation.py), all trained under the *same* budget
# (epochs=25, early_stop=5) so this is an apples-to-apples comparison -- each checkpoint
# lives at records/DFB/fusion_ablation/models_<mode>/best.pth.
FUSION_MODES = ['addition', 'concatenation', 'attention', 'dual_cross_attention']

# A sensible default spread of held-out datasets: Celeb-DF-v2 and DeepFakeDetection are
# genuinely different face pools from FF++, UADFV is a different (older, lower-res) source
# entirely. DFDC is available too (pass --targets ...,dfdc) but left out of the default
# since it's the largest and slowest to index.
DEFAULT_TARGETS = ['celebdfv2', 'deepfakedetection', 'uadfv']


def _model_kwargs(fusion_mode: str) -> dict:
    return dict(
        s3_weight=0.3, face_size=112, use_spatial=True, use_noise=True, use_temporal=True,
        fusion_mode=fusion_mode, use_localization_head=True, use_identification_head=True,
        use_landmark_attention=True,
    )


def _run_target(project_root: Path, cfg: dict, ablation_dir: Path, out_dir: Path, target: str, compressions: str) -> list:
    ffpp_cfg = cfg.get('dfb', {}).get('ffpp', {})
    loader_cfg = {
        'n_frames': ffpp_cfg.get('n_frames', 32), 'face_size': ffpp_cfg.get('face_size', 112),
        'img_size': ffpp_cfg.get('img_size', 224), 'batch_size': cfg.get('train', {}).get('batch_size', 4),
        'num_workers': cfg.get('train', {}).get('num_workers', 4),
    }
    compression = None if target in ('uadfv', 'dfdc', 'celebdfv2') else compressions
    items = _build_items(project_root, cfg, target, compression)
    loader = get_dfb_eval_loader(items, loader_cfg)
    n_real = sum(1 for it in items if it['label'] == 0)
    n_fake = sum(1 for it in items if it['label'] == 1)
    print(f'\n[{target}] real={n_real} fake={n_fake} total={len(items)}')

    rows = []
    for mode in FUSION_MODES:
        model_path = ablation_dir / f'models_{mode}' / 'best.pth'
        if not model_path.exists():
            print(f'  [skip] {mode}: no checkpoint at {model_path} yet')
            continue
        print(f'  ===== fusion_mode: {mode} on {target} =====')
        result = evaluate_dfb_loader(model_path, loader, out_dir / f'{mode}_on_{target}', f'dfb_{target}',
                                      model_kwargs=_model_kwargs(mode))
        m = result['metrics']
        row = {'fusion_mode': mode, 'target': target, 'auc': m['auc'], 'accuracy': m['accuracy'],
               'f1': m.get('f1', float('nan')), 'mask_iou': m.get('mask_iou', float('nan'))}
        rows.append(row)
        print(f"    -> {mode}: AUC={m['auc']:.4f} Acc={m['accuracy']:.4f} F1={row['f1']:.4f}")

    if rows:
        csv_path = out_dir / f'fusion_cross_eval_{target}.csv'
        existing = {}
        if csv_path.exists():
            with open(csv_path, newline='') as fh:
                for r in csv.DictReader(fh):
                    existing[r['fusion_mode']] = r
        for row in rows:
            existing[row['fusion_mode']] = row
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['fusion_mode', 'target', 'auc', 'accuracy', 'f1', 'mask_iou'])
            w.writeheader()
            for mode in FUSION_MODES:
                if mode in existing:
                    w.writerow(existing[mode])
        print(f'  Wrote {csv_path}')
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Fusion-strategy cross-domain generalization comparison: evaluates every fusion_ablation '
                    'checkpoint (addition/concatenation/attention/dual_cross_attention, all trained under the '
                    'identical epochs=25/early_stop=5 budget) against one or more held-out DeepFakeBench targets.')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--targets', default=None,
                         help=f"Comma-separated subset of {TARGETS}. Default: {DEFAULT_TARGETS}.")
    parser.add_argument('--compressions', default='c40', help="Compression used to build the eval index, where applicable.")
    args = parser.parse_args()

    targets = DEFAULT_TARGETS
    if args.targets:
        wanted = {t.strip() for t in args.targets.split(',')}
        targets = [t for t in TARGETS if t in wanted]
        if not targets:
            raise ValueError(f'--targets={args.targets!r} matched none of {TARGETS}')

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)

    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    ablation_dir = records_dir / 'fusion_ablation'
    out_dir = records_dir / 'fusion_cross_eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for target in targets:
        all_rows += _run_target(PROJECT_ROOT, cfg, ablation_dir, out_dir, target, args.compressions)

    if not all_rows:
        print('\nNo checkpoints available yet -- nothing to compare.')
        return

    # Combined summary across all targets run this invocation, one row per (fusion_mode, target).
    summary_path = out_dir / 'fusion_cross_eval_summary.csv'
    existing = {}
    if summary_path.exists():
        with open(summary_path, newline='') as fh:
            for r in csv.DictReader(fh):
                existing[(r['fusion_mode'], r['target'])] = r
    for row in all_rows:
        existing[(row['fusion_mode'], row['target'])] = row
    with open(summary_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['fusion_mode', 'target', 'auc', 'accuracy', 'f1', 'mask_iou'])
        w.writeheader()
        for key in sorted(existing, key=lambda k: (k[1], FUSION_MODES.index(k[0]) if k[0] in FUSION_MODES else 99)):
            w.writerow(existing[key])
    print(f'\nWrote {summary_path}')


if __name__ == '__main__':
    main()
