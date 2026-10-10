from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_dataset import get_dfb_eval_loader
from forensicmtf.dfb_runner import _eval_model_kwargs, _load_or_build_ffpp_index, resolve_path
from forensicmtf.evaluation.dfb_eval import evaluate_dfb_loader

# T/F/S stream-contribution ablation, done by *blocking* (zeroing the raw input of) one or
# more streams of an already-trained, unmodified headline checkpoint at eval time -- no
# retraining. T=temporal (Stream1, video clip), F=frequency/noise (Stream2: BayarConv2d +
# FFT + Haar-DWT, face crop), S=spatial (Stream3: face crop + landmark heatmap).
# Every combo still runs through the model's normally-trained fusion code path (see the
# block_streams docstring in evaluate_dfb_loader) -- this measures the trained model's
# reliance on each stream, not what a model retrained without that stream would score.
# name -> set of streams to BLOCK (zero out)
COMBOS = {
    'Temporal only':  {'frequency', 'spatial'},
    'Frequency only': {'temporal', 'spatial'},
    'Spatial only':   {'temporal', 'frequency'},
    'T+F (no S)':     {'spatial'},
    'T+S (no F)':     {'frequency'},
    'F+S (no T)':     {'temporal'},
    'Full model':     set(),
}


def main():
    parser = argparse.ArgumentParser(
        description='T/F/S stream-contribution ablation via inference-time input blocking on the existing '
                    'headline checkpoints (records/DFB/models_ffpp_<c23,c40>/best.pth) -- no retraining.')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--compressions', default='c23,c40', help='Comma-separated compressions to evaluate, matching existing headline checkpoints.')
    args = parser.parse_args()
    compressions = [c.strip() for c in args.compressions.split(',') if c.strip()]

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)
    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    out_root = records_dir / 'stream_block_ablation'
    out_root.mkdir(parents=True, exist_ok=True)

    # rows[combo_name][compression] = {'auc':..., 'accuracy':...}
    rows: dict[str, dict[str, dict]] = {name: {} for name in COMBOS}

    for comp in compressions:
        run_cfg = dict(cfg)
        run_cfg['dfb'] = dict(cfg.get('dfb', {}))
        run_cfg['dfb']['ffpp'] = dict(cfg['dfb']['ffpp'])
        run_cfg['dfb']['ffpp']['compressions'] = [comp]

        index_by_split = _load_or_build_ffpp_index(PROJECT_ROOT, run_cfg, smoke=False)
        ffpp_cfg = run_cfg['dfb']['ffpp']
        loader_cfg = {
            'n_frames': ffpp_cfg.get('n_frames', 32), 'face_size': ffpp_cfg.get('face_size', 112),
            'img_size': ffpp_cfg.get('img_size', 224), 'batch_size': cfg.get('train', {}).get('batch_size', 4),
            'num_workers': cfg.get('train', {}).get('num_workers', 4),
        }
        test_loader = get_dfb_eval_loader(index_by_split['test'], loader_cfg)

        model_path = records_dir / f'models_ffpp_{comp}' / 'best.pth'
        if not model_path.exists():
            print(f'[skip] {comp}: no headline checkpoint at {model_path}')
            continue
        model_kwargs = _eval_model_kwargs(dict(cfg.get('train', {})))

        for name, blocked in COMBOS.items():
            print(f'\n===== {comp}: {name} (blocking {sorted(blocked) or "nothing"}) =====')
            tag = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
            out_dir = out_root / 'eval' / f'{comp}_{tag}'
            result = evaluate_dfb_loader(model_path, test_loader, out_dir, f'dfb_ffpp_{tag}',
                                          model_kwargs=model_kwargs, block_streams=blocked)
            m = result['metrics']
            rows[name][comp] = {'auc': m['auc'], 'accuracy': m['accuracy']}
            print(f"  -> AUC={m['auc']:.4f} Acc={m['accuracy']:.4f}")

    csv_path = out_root / 'table_stream_block.csv'
    header = ['variant'] + [f'{c}_auc' for c in compressions] + [f'{c}_accuracy' for c in compressions]
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for name in COMBOS:
            vals = rows[name]
            w.writerow([name] + [vals.get(c, {}).get('auc', '') for c in compressions]
                       + [vals.get(c, {}).get('accuracy', '') for c in compressions])
    print(f'\nWrote {csv_path}')


if __name__ == '__main__':
    main()
