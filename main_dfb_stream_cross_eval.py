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
from main_dfb_cross_eval import TARGETS, _build_items  # noqa: E402

# Justifies the Spatial stream the same way main_dfb_stream_robustness.py justifies the
# Frequency stream -- not via clean in-domain AUC (where it showed ~nothing, see the T+S-no-F
# finding), but via a property clean-data AUC can't see. For Frequency that property was
# robustness under perturbation; for Spatial the natural parallel is cross-domain
# generalization: landmark-anchored, geometry-aware features should transfer across datasets
# better than a purely learned spatiotemporal CNN tuned to FF++'s specific artifact
# statistics. Compares Temporal-only, T+S (no F), and the Full model zero-shot on three
# held-out datasets, using the same targets as main_dfb_fusion_cross_eval.py.
CHECKPOINTS = {
    'Temporal only': ('stream_ablation/models_{comp}_temporal_only', dict(use_temporal=True, use_noise=False, use_spatial=False)),
    'T+S (no F)':    ('stream_ablation/models_{comp}_tpluss_no_f',   dict(use_temporal=True, use_noise=False, use_spatial=True)),
    'Full model':    ('models_ffpp_{comp}',                          dict(use_temporal=True, use_noise=True,  use_spatial=True)),
}

DEFAULT_TARGETS = ['celebdfv2', 'deepfakedetection', 'uadfv']


def _model_kwargs(stream_kwargs: dict) -> dict:
    return dict(s3_weight=0.3, face_size=112, fusion_mode='dual_cross_attention',
                use_localization_head=True, use_identification_head=True, use_landmark_attention=True,
                **stream_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Cross-domain generalization comparison: does adding the Spatial stream to Temporal help '
                    'zero-shot transfer, even though it does not help on clean in-domain FF++ AUC?')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--compressions', default='c23', help="Which stream-ablation checkpoints to use ('c23' or 'c40').")
    parser.add_argument('--targets', default=None, help=f"Comma-separated subset of {TARGETS}. Default: {DEFAULT_TARGETS}.")
    args = parser.parse_args()
    comp = args.compressions

    targets = DEFAULT_TARGETS
    if args.targets:
        wanted = {t.strip() for t in args.targets.split(',')}
        targets = [t for t in TARGETS if t in wanted]
        if not targets:
            raise ValueError(f'--targets={args.targets!r} matched none of {TARGETS}')

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    ffpp_cfg = cfg.get('dfb', {}).get('ffpp', {})
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    out_dir = records_dir / 'stream_cross_eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    loader_cfg_base = {
        'n_frames': ffpp_cfg.get('n_frames', 32), 'face_size': ffpp_cfg.get('face_size', 112),
        'img_size': ffpp_cfg.get('img_size', 224), 'batch_size': 2, 'num_workers': 2,
    }

    all_rows = []
    for target in targets:
        compression = None if target in ('uadfv', 'dfdc', 'celebdfv2') else comp
        items = _build_items(PROJECT_ROOT, cfg, target, compression)
        loader = get_dfb_eval_loader(items, loader_cfg_base)
        n_real = sum(1 for it in items if it['label'] == 0)
        n_fake = sum(1 for it in items if it['label'] == 1)
        print(f'\n[{target}] real={n_real} fake={n_fake} total={len(items)}')

        for name, (rel_path, stream_kwargs) in CHECKPOINTS.items():
            model_path = records_dir / rel_path.format(comp=comp) / 'best.pth'
            if not model_path.exists():
                print(f'  [skip] {name}: no checkpoint at {model_path} yet')
                continue
            print(f'  ===== {name} on {target} =====')
            result = evaluate_dfb_loader(model_path, loader, out_dir / f'{name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")}_on_{target}',
                                          f'dfb_{target}', model_kwargs=_model_kwargs(stream_kwargs))
            m = result['metrics']
            row = {'checkpoint': name, 'target': target, 'auc': m['auc'], 'accuracy': m['accuracy'], 'f1': m.get('f1', float('nan'))}
            all_rows.append(row)
            print(f"    -> AUC={m['auc']:.4f} Acc={m['accuracy']:.4f}")

    if not all_rows:
        print('\nNo checkpoints available yet.')
        return

    csv_path = out_dir / 'stream_cross_eval_summary.csv'
    existing = {}
    if csv_path.exists():
        with open(csv_path, newline='') as fh:
            for r in csv.DictReader(fh):
                existing[(r['checkpoint'], r['target'])] = r
    for row in all_rows:
        existing[(row['checkpoint'], row['target'])] = row
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['checkpoint', 'target', 'auc', 'accuracy', 'f1'])
        w.writeheader()
        for key in sorted(existing, key=lambda k: (k[1], list(CHECKPOINTS).index(k[0]) if k[0] in CHECKPOINTS else 99)):
            w.writerow(existing[key])
    print(f'\nWrote {csv_path}')


if __name__ == '__main__':
    main()
