from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_index import (
    build_dfb_celebdf_index,
    build_dfb_dfdc_index,
    build_dfb_ffpp_family_index,
    build_dfb_ffpp_index,
    build_dfb_uadfv_index,
)
from forensicmtf.dfb_runner import resolve_path, run_dfb_flat_eval

TARGETS = ['faceshifter', 'deepfakedetection', 'uadfv', 'dfdc', 'celebdfv2']


def _build_items(project_root: Path, cfg: dict, target: str, compression: str | None) -> list:
    if target in ('faceshifter', 'deepfakedetection'):
        ffpp_root = resolve_path(project_root, cfg['paths']['dfb_root']) / 'FaceForensics++'
        if target == 'faceshifter':
            # Same real/fake pairing convention as the 4 core methods (splits/test.json) -
            # this model's training fake_subsets never included FaceShifter, so the FF++
            # test split is a leak-free zero-shot cross-method eval here.
            return build_dfb_ffpp_index(ffpp_root, 'test', compressions=[compression], fake_subsets=['FaceShifter'])
        # DeepFakeDetection pairs with a disjoint actor real-video pool the model never
        # saw at all during training (only youtube reals were used) - no split needed,
        # the whole family is zero-shot.
        return build_dfb_ffpp_family_index(ffpp_root, compression, 'DeepFakeDetection', real_dir_name='actors')
    if target == 'uadfv':
        uadfv_root = resolve_path(project_root, cfg['paths']['uadfv_root'])
        return build_dfb_uadfv_index(uadfv_root)
    if target == 'dfdc':
        dfdc_root = resolve_path(project_root, cfg['paths']['dfdc_root'])
        return build_dfb_dfdc_index(dfdc_root)
    if target == 'celebdfv2':
        # Celeb-DF-v2 has the exact same on-disk layout as v1 (Celeb-real/YouTube-real/
        # Celeb-synthesis + List_of_testing_videos.txt), so the same index builder applies
        # unmodified - only the root path differs. Only the official test split is used.
        celebdf_v2_root = resolve_path(project_root, cfg['paths']['celebdf_v2_dfb_root'])
        return build_dfb_celebdf_index(celebdf_v2_root)['test']
    raise ValueError(f'Unknown target: {target!r} (choices: {TARGETS})')


def main():
    parser = argparse.ArgumentParser(
        description='ForensicMTF DFB extra cross-domain/cross-method eval targets '
                    '(FaceShifter, DeepFakeDetection, UADFV) against an already-trained checkpoint.')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--target', required=True, choices=TARGETS)
    parser.add_argument('--compressions', default='c40',
                        help="Which trained checkpoint to evaluate, matching the --compressions used at train "
                             "time (e.g. 'c40' or 'c23') - resolves to records/DFB/models_ffpp_<compressions>/best.pth. "
                             "Ignored for --target uadfv only insofar as UADFV itself has no compression variants; "
                             "the checkpoint selection still applies.")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)

    train_variant = f'ffpp_{args.compressions}'
    compression = None if args.target in ('uadfv', 'dfdc', 'celebdfv2') else args.compressions
    items = _build_items(PROJECT_ROOT, cfg, args.target, compression)
    n_real = sum(1 for it in items if it['label'] == 0)
    n_fake = sum(1 for it in items if it['label'] == 1)
    if not items:
        raise RuntimeError(f'No items found for target={args.target!r} - check the dataset path in the config.')
    print(f'[{args.target}] checkpoint=models_{train_variant} real={n_real} fake={n_fake} total={len(items)}')

    result = run_dfb_flat_eval(PROJECT_ROOT, cfg, items, f'dfb_{args.target}', train_variant)
    print(f"\n{args.target} vs. {train_variant}: {result['metrics']}")
    print(f"outputs: {result['out_dir']}")


if __name__ == '__main__':
    main()
