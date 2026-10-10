from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from forensicmtf.config import ensure_records_layout, load_config
from forensicmtf.data.dfb_index import build_dfb_celebdf_index
from forensicmtf.dfb_runner import _load_or_build_ffpp_index, resolve_path, run_dfb_ablation_variant

TABLE_HEADERS = ['Variant', 'FF++ AUC', 'FF++ ACC', 'FF++ F1', 'Mask IoU', 'ID Acc', 'CelebDF AUC', 'CelebDF ACC', 'CelebDF F1']


def _fmt(row: dict, key: str) -> str:
    val = row.get(key)
    return f'{val:.4f}' if isinstance(val, (int, float)) and val == val else 'nan'


def _write_tables(rows: list, out_dir: Path) -> None:
    if not rows:
        print('No ablation results to write.')
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / 'table_mlg_mtf.csv'
    fieldnames = sorted({k for row in rows for k in row.keys()}, key=lambda k: (k != 'name', k != 'variant', k))
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    col_rows = [
        [
            row['name'], _fmt(row, 'ffpp_auc'), _fmt(row, 'ffpp_accuracy'), _fmt(row, 'ffpp_f1'),
            _fmt(row, 'ffpp_mask_iou'), _fmt(row, 'ffpp_id_acc'),
            _fmt(row, 'celebdf_auc'), _fmt(row, 'celebdf_accuracy'), _fmt(row, 'celebdf_f1'),
        ]
        for row in rows
    ]
    widths = [max(len(h), *(len(r[i]) for r in col_rows)) for i, h in enumerate(TABLE_HEADERS)]
    txt_path = out_dir / 'table_mlg_mtf.txt'
    with open(txt_path, 'w') as fh:
        fh.write('Table: MLG-MTF ablation - mask localization + manipulation-method identification + landmark-guided attention\n\n')
        fh.write(' | '.join(h.ljust(w) for h, w in zip(TABLE_HEADERS, widths)) + '\n')
        fh.write('-+-'.join('-' * w for w in widths) + '\n')
        for r in col_rows:
            fh.write(' | '.join(c.ljust(w) for c, w in zip(r, widths)) + '\n')

    print(f'\nWrote {csv_path}')
    print(f'Wrote {txt_path}')
    print()
    with open(txt_path) as fh:
        print(fh.read())


def main():
    parser = argparse.ArgumentParser(description='ForensicMTF DFB MLG-MTF ablation sweep (config_dfb.yaml: ablation.mlg_mtf)')
    parser.add_argument('--config', default='config_dfb.yaml')
    parser.add_argument('--variants', default=None,
                        help='Comma-separated dataset_variant slugs to run (default: all variants listed in the config)')
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    ensure_records_layout(PROJECT_ROOT, cfg)

    variants = cfg.get('ablation', {}).get('mlg_mtf', [])
    if not variants:
        raise ValueError("No variants found under config['ablation']['mlg_mtf'] in the config file.")
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(',')}
        variants = [v for v in variants if v.get('dataset_variant') in wanted]
        if not variants:
            raise ValueError(f'No configured variants matched --variants={args.variants!r}.')

    sweep_cfg = cfg.get('ablation', {}).get('sweep', {})
    print(f"Running {len(variants)} ablation variant(s), epochs={sweep_cfg.get('epochs', 25)} "
          f"early_stop={sweep_cfg.get('early_stop', 5)} (override via ablation.sweep in the config)")

    # Both the FF++ index and Celeb-DF-v1 index are shared read-only across all variants -
    # build once, reuse for every variant's train/eval instead of rebuilding per variant.
    index_by_split = _load_or_build_ffpp_index(PROJECT_ROOT, cfg, smoke=False)
    celebdf_root = resolve_path(PROJECT_ROOT, cfg['paths']['celebdf_dfb_root'])
    celebdf_index = build_dfb_celebdf_index(celebdf_root)

    rows = []
    for variant in variants:
        row = run_dfb_ablation_variant(PROJECT_ROOT, cfg, variant, index_by_split, celebdf_index, sweep_cfg)
        rows.append(row)
        print(f"  -> {row['name']}: FF++ AUC={_fmt(row, 'ffpp_auc')} IoU={_fmt(row, 'ffpp_mask_iou')} "
              f"CelebDF AUC={_fmt(row, 'celebdf_auc')}")

    records_dir = resolve_path(PROJECT_ROOT, cfg.get('project', {}).get('records_dir', 'records/DFB'))
    _write_tables(rows, records_dir / 'ablation')


if __name__ == '__main__':
    main()
