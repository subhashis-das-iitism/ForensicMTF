from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_eval_loader
from forensicmtf.dfb_runner import _ffpp_cfg, resolve_path
from forensicmtf.models.forensicmtf import ForensicMTF
from main_dfb_robustness import PERTURBATIONS, _load_or_build_ffpp_index_for_compression, _run_one  # noqa: E402

# Does the Frequency/Noise branch earn its keep under *degraded* input, even though it showed
# no marginal clean-data AUC benefit (see the T+S-no-F finding in the fusion/stream report)?
# Classical forensic high-pass filters (BayarConv2d, Haar-DWT) are motivated by robustness to
# exactly this kind of degradation, not clean-data accuracy -- this is the right place to look
# for their value. Compares Temporal-only, T+F (no S), and the Full model's degradation curve
# on the same perturbation suite as main_dfb_robustness.py.
CHECKPOINTS = {
    'Temporal only': ('stream_ablation/models_{comp}_temporal_only', dict(use_temporal=True, use_noise=False, use_spatial=False)),
    'T+F (no S)':    ('stream_ablation/models_{comp}_tplusf_no_s',   dict(use_temporal=True, use_noise=True,  use_spatial=False)),
    'Full model':    ('models_ffpp_{comp}',                          dict(use_temporal=True, use_noise=True,  use_spatial=True)),
}


def main():
    parser = argparse.ArgumentParser(
        description='Robustness-under-perturbation comparison: does adding the Frequency/Noise stream to '
                    'Temporal help once the input is degraded, even though it does not help on clean data?')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--compressions', default='c40')
    parser.add_argument('--batch_size', type=int, default=2,
                         help='Kept small (default 2, vs. the usual training batch_size=4) so this can safely '
                              'run concurrently on the same GPU as an in-progress training job without risking OOM.')
    args = parser.parse_args()
    comp = args.compressions

    torch.manual_seed(42)  # match main_dfb_robustness.py's fixed seed for comparable noise draws
    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))

    index_by_split = _load_or_build_ffpp_index_for_compression(PROJECT_ROOT, cfg, comp)
    ffpp_cfg = _ffpp_cfg(cfg)
    loader_cfg = {
        'n_frames': ffpp_cfg.get('n_frames', 32), 'face_size': ffpp_cfg.get('face_size', 112),
        'img_size': ffpp_cfg.get('img_size', 224), 'batch_size': args.batch_size,
        'num_workers': 2,
    }
    loader = get_dfb_eval_loader(index_by_split['test'], loader_cfg)

    all_rows = []
    for name, (rel_path, stream_kwargs) in CHECKPOINTS.items():
        model_path = records_dir / rel_path.format(comp=comp) / 'best.pth'
        if not model_path.exists():
            print(f'[skip] {name}: no checkpoint at {model_path} yet')
            continue
        print(f'\n===== {name} ({model_path}) =====')
        model_kwargs = dict(s3_weight=0.3, face_size=112, fusion_mode='dual_cross_attention',
                             use_localization_head=True, use_identification_head=True, use_landmark_attention=True,
                             **stream_kwargs)
        model = ForensicMTF(pretrained=False, **model_kwargs).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) and ('ema' in ckpt or 'model' in ckpt) else ckpt
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        clean_auc = None
        for pname in PERTURBATIONS:
            row = _run_one(model, loader, device, pname)
            row['checkpoint'] = name
            if pname == 'clean':
                clean_auc = row['auc']
            row['delta_auc'] = round((clean_auc - row['auc']) if clean_auc is not None else 0.0, 4)
            all_rows.append(row)
            print(f"  {pname:<16}: AUC={row['auc']:.4f} delta={row['delta_auc']:+.4f} Acc={row['accuracy']:.4f}")

    if not all_rows:
        print('\nNo checkpoints available yet.')
        return

    out_dir = records_dir / 'stream_robustness'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'stream_robustness_{comp}.csv'
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['checkpoint', 'perturbation', 'auc', 'delta_auc', 'eer', 'accuracy', 'threshold', 'n'])
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
