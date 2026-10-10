from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_eval_loader, get_dfb_loaders
from forensicmtf.data.dfb_index import build_dfb_ffpp_family_index_stratified
from forensicmtf.dfb_runner import _build_model, _eval_model_kwargs, resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader
from forensicmtf.training.dfb_core import train_dfb

# In-domain (trained-and-tested-on-DeepFakeDetection) checkpoint, so Table VIII's
# DeepFakeDetection row can be compared against MVIM's own in-domain-trained number under
# a matching protocol, rather than our existing zero-shot (trained on FF++ only) number.
# Same architecture/hyperparams as the FF++ headline model (train.* in config_dfb.yaml) so
# the only difference is the training data. No official train/val/test split exists for
# this FF++ family (unlike the four core methods), so this uses a stratified random split
# (build_dfb_ffpp_family_index_stratified) -- disclosed as a limitation, same as the
# Celeb-DF-v1 in-domain run.


def main():
    parser = argparse.ArgumentParser(description='In-domain (train+test on DeepFakeDetection) checkpoint, matching MVIM Table III protocol.')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--compressions', default='c40', help="Which compression to train/test on ('c40' or 'c23').")
    parser.add_argument('--task', default='all', choices=['train', 'evaluate', 'all'])
    parser.add_argument('--mask_loss_weight', type=float, default=None,
                         help='Override train.mask_loss_weight from the config for this run only '
                              '(e.g. to recover from a collapsed localization head under the default weight).')
    parser.add_argument('--tag', default='',
                         help="Suffix for save_dir/metrics_path/eval out_dir (e.g. '_maskw06'), so an "
                              "experimental override doesn't overwrite the existing verified checkpoint/results.")
    parser.add_argument('--resume', default=None,
                         help='Path to a checkpoints/epoch_NNN.pth to resume from (full optimizer/scheduler state, '
                              'unlike best.pth which is EMA-weights-only).')
    parser.add_argument('--epochs', type=int, default=None,
                         help='Override train.epochs (total epoch count, not additional epochs) -- '
                              'e.g. resume from epoch 50 with --epochs 60 to train 10 more.')
    parser.add_argument('--init_model_path', default=None,
                         help='Warm-start weights (e.g. records/DFB/models_ffpp_c40/best.pth) loaded via a '
                              'shape-tolerant state_dict merge before training starts -- unlike --resume, this '
                              'starts a fresh optimizer/scheduler/epoch-0 run, just from pretrained weights. '
                              'Ignored if --resume is also set (resume takes priority in train_dfb).')
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))

    ffpp_root = resolve_path(PROJECT_ROOT, cfg['paths']['dfb_root']) / 'FaceForensics++'
    index_by_split = build_dfb_ffpp_family_index_stratified(
        ffpp_root, args.compressions, 'DeepFakeDetection', real_dir_name='actors')
    print(f"train={len(index_by_split['train'])} val={len(index_by_split['val'])} test={len(index_by_split['test'])}")

    train_cfg = dict(cfg.get('train', {}))
    if args.mask_loss_weight is not None:
        train_cfg['mask_loss_weight'] = args.mask_loss_weight
    if args.resume is not None:
        train_cfg['resume'] = args.resume
    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs
    if args.init_model_path is not None:
        train_cfg['init_model_path'] = args.init_model_path
    save_dir = records_dir / f'models_deepfakedetection_indomain_{args.compressions}{args.tag}'
    train_cfg['save_dir'] = str(save_dir)
    train_cfg['metrics_path'] = str(records_dir / 'results' / f'metrics_deepfakedetection_indomain_{args.compressions}{args.tag}.csv')
    train_cfg['metric_names'] = ['real', 'DeepFakeDetection']

    loader_cfg = {
        'n_frames': train_cfg.get('n_frames', 32), 'face_size': train_cfg.get('face_size', 112),
        'img_size': train_cfg.get('img_size', 224), 'batch_size': train_cfg.get('batch_size', 4),
        'num_workers': train_cfg.get('num_workers', 4), 'augmentations': train_cfg.get('augmentations'),
    }

    if args.task in ('train', 'all'):
        train_dl, val_dl, test_dl = get_dfb_loaders(index_by_split, loader_cfg)
        model = _build_model(train_cfg)
        train_dfb(train_cfg, train_dl, val_dl, test_dl, model)

    if args.task in ('evaluate', 'all'):
        test_loader = get_dfb_eval_loader(index_by_split['test'], loader_cfg)
        model_path = save_dir / 'best.pth'
        out_dir = records_dir / 'eval' / f'dfb_deepfakedetection_indomain_{args.compressions}{args.tag}'
        model_kwargs = _eval_model_kwargs(train_cfg)
        result = evaluate_dfb_loader(model_path, test_loader, out_dir, 'dfb_deepfakedetection_indomain', model_kwargs=model_kwargs)
        print(f"\nDeepFakeDetection in-domain test: {result['metrics']}")
        print(f"outputs: {result['out_dir']}")


if __name__ == '__main__':
    main()
