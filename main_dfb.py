from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.dfb_runner import (
    build_dfb_ffpp_index_by_split,
    run_dfb_cross_domain_evaluation,
    run_dfb_evaluation,
    run_dfb_training,
)


def main():
    parser = argparse.ArgumentParser(description='ForensicMTF DeepFakeBench (DFB) runner')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--task', default=None, choices=['build_index', 'train', 'evaluate', 'cross_domain_eval', 'all'])
    parser.add_argument('--smoke', action='store_true', help='Use a small truncated index + short training run for pipeline validation.')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed python/numpy/torch for a reproducible re-run (e.g. for a confidence-interval '
                            "sweep) and suffix save_dir/metrics_path with '_seed<N>' so it never overwrites the "
                            'existing unseeded headline checkpoint/metrics for the same --compressions variant.')
    parser.add_argument('--compressions', default=None,
                       help="Comma-separated override for dfb.ffpp.compressions, e.g. 'c40' or 'c23,c40'. "
                            "Also determines the save_dir/metrics_path/index variant name (ffpp_<compressions joined by _>), "
                            "so c23 and c40 runs land in separate records/DFB/models_ffpp_c23 vs models_ffpp_c40 directories.")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    if args.compressions:
        cfg.setdefault('dfb', {}).setdefault('ffpp', {})['compressions'] = [c.strip() for c in args.compressions.split(',') if c.strip()]
    ensure_records_layout(PROJECT_ROOT, cfg)
    workflow = cfg.get('workflow', {})
    task = args.task or workflow.get('task', 'train')

    if task in ('build_index', 'all'):
        build_dfb_ffpp_index_by_split(PROJECT_ROOT, cfg, smoke=args.smoke)
    if task in ('train', 'all'):
        run_dfb_training(PROJECT_ROOT, cfg, smoke=args.smoke, resume=args.resume, seed=args.seed)
    if task in ('evaluate', 'all'):
        run_dfb_evaluation(PROJECT_ROOT, cfg, smoke=args.smoke, seed=args.seed)
    if task == 'cross_domain_eval':
        run_dfb_cross_domain_evaluation(PROJECT_ROOT, cfg, smoke=args.smoke)


if __name__ == '__main__':
    main()
