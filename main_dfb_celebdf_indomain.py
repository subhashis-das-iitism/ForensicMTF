from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_eval_loader, get_dfb_loaders
from forensicmtf.data.dfb_index import build_dfb_celebdf_index, build_dfb_celebdf_index_stratified
from forensicmtf.dfb_runner import _build_model, _eval_model_kwargs, resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader
from forensicmtf.training.dfb_core import train_dfb

# In-domain (trained-and-tested-on-Celeb-DF-v1) checkpoint, to fill in the CUTA full-sweep
# table's Celeb-DF column with a genuine matching-protocol number, rather than leaving it
# blank or substituting our zero-shot cross-domain number (which belongs in Tables V/VI, a
# different, harder setting). Same architecture/hyperparams as the FF++ headline model
# (train.* in config_dfb.yaml) so the only difference is the training data. Test split is
# the official List_of_testing_videos.txt (identity-disjoint from train/val); train/val use
# a stratified random split, NOT identity-disjoint, since Celeb-DF-v1's synthesis pairs
# interconnect nearly all identities into one union-find component, making an
# identity-disjoint *and* class-balanced val split infeasible (see
# build_dfb_celebdf_index_stratified's docstring) -- disclosed as a limitation.


def main():
    parser = argparse.ArgumentParser(description='In-domain (train+test on Celeb-DF-v1) checkpoint, matching CUTA Table I protocol.')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--task', default='all', choices=['train', 'evaluate', 'all'])
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))

    celebdf_root = resolve_path(PROJECT_ROOT, cfg['paths']['celebdf_dfb_root'])
    index_by_split = build_dfb_celebdf_index_stratified(celebdf_root)
    print(f"train={len(index_by_split['train'])} val={len(index_by_split['val'])} test={len(index_by_split['test'])}")

    train_cfg = dict(cfg.get('train', {}))
    save_dir = records_dir / 'models_celebdf_indomain'
    train_cfg['save_dir'] = str(save_dir)
    train_cfg['metrics_path'] = str(records_dir / 'results' / 'metrics_celebdf_indomain.csv')
    train_cfg['metric_names'] = ['real', 'fake']

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
        out_dir = records_dir / 'eval' / 'dfb_celebdf_indomain'
        model_kwargs = _eval_model_kwargs(train_cfg)
        result = evaluate_dfb_loader(model_path, test_loader, out_dir, 'dfb_celebdf_indomain', model_kwargs=model_kwargs)
        print(f"\nCeleb-DF-v1 in-domain test: {result['metrics']}")
        print(f"outputs: {result['out_dir']}")


if __name__ == '__main__':
    main()
