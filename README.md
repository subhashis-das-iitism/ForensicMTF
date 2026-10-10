# ForensicMTF-DFB

DeepFakeBench-native deepfake detection: trains and evaluates directly against
DeepFakeBench's pre-extracted frames/landmarks/masks (FaceForensics++ c23/c40,
Celeb-DF-v1), with an optional path for building a cache from raw video instead.
Model: `ForensicMTF` (`src/forensicmtf/models/forensicmtf.py`) - a three-stream detector (temporal / noise-frequency / landmark-guided spatial) with mask-supervised
forgery localization and manipulation-method identification heads.

## Environment

```bash
pip install -r requirements.txt
```

## Dataset paths

Set `paths.dfb_root` and `paths.celebdf_dfb_root` in `config_dfb.yaml` to point at your
DeepFakeBench `rgb/` root (containing `FaceForensics++/` and `Celeb-DF-v1/`).

All `paths.*` entries default to `${DFB_DATA_ROOT:-<checked-in default>}`, so on a new
machine you can override every dataset root at once without editing the config:

```bash
DFB_DATA_ROOT=/path/to/deepfakebench/rgb python3 main_dfb.py --config config_dfb.yaml --task build_index
```

## Usage

```bash
# Build the FF++ sample index (cached under records/DFB/index/)
python3 main_dfb.py --config config_dfb.yaml --task build_index

# Train
python3 main_dfb.py --config config_dfb.yaml --task train

# Evaluate on the FF++ test split
python3 main_dfb.py --config config_dfb.yaml --task evaluate

# Cross-domain evaluate the same checkpoint on Celeb-DF-v1's official test split
python3 main_dfb.py --config config_dfb.yaml --task cross_domain_eval
```

Compression is set via `dfb.ffpp.compressions` in the config, or overridden per-run:

```bash
python3 main_dfb.py --config config_dfb.yaml --compressions c23 --task train
```

`--smoke` truncates the index and shortens training for a fast pipeline sanity check
(`--task build_index/train/evaluate/cross_domain_eval --smoke`).

To train both compressions end-to-end (build_index -> train -> evaluate ->
cross_domain_eval, per compression), see `scripts/run_dfb_full_training.sh`.

### Ablation sweep

```bash
python3 main_dfb_ablation.py --config config_dfb.yaml
```

Runs the `ablation.mlg_mtf` variants from the config (base vs. +landmark-attention vs.
+localization vs. +identification), each trained for a reduced epoch/patience budget
(`ablation.sweep` in the config), then writes a combined comparison table to
`records/DFB/ablation/table_mlg_mtf.{csv,txt}`.

### Stream ablations

`main_dfb_stream_ablation.py` (and `_multiframe` for the multi-frame `ForensicMTFMultiFrame`
variant) retrains with subsets of the three streams; `main_dfb_stream_block_eval.py`,
`main_dfb_stream_cross_eval.py` and `main_dfb_stream_robustness.py` probe the trained
checkpoints. Pass `--seed N` to `main_dfb.py` for a seeded, reproducible re-run.

### Raw-video preprocessing

If you need to build a training cache from raw `.mp4` videos instead of DeepFakeBench's
pre-extracted assets, `src/forensicmtf/data/cache_builder.py` samples frames uniformly
(`uniform_select`) and crops faces; `build_cache(...)` is the entry point. This isn't
wired to a CLI task currently (the DFB pipeline above never needs it), but the module is
self-contained and importable.

## Outputs

Everything lands under `records/DFB/`. The results reported in the paper are checked in
(metrics CSVs, plots, tables and logs); model checkpoints (`*.pth`) and the sample index
are not, because of their size and machine-specific paths.

- `index/` - cached sample indices (JSON)
- `models_<variant>/` - checkpoints (`best.pth` + per-epoch)
- `results/metrics_<variant>.csv` - per-epoch train/val metrics
- `eval/<dataset>_<variant>/` - `metrics_summary.csv`, `per_video_scores.csv`, ROC/confusion/score-distribution plots, `localization_examples.png`
- `ablation/`, `fusion_ablation/`, `stream_ablation*/`, `freqmask_ablation/`, `s3_sweep/` - ablation outputs and comparison tables
- `robustness/`, `stream_robustness/`, `cross_manipulation/`, `fusion_cross_eval/` - robustness and generalization results
- `model_stats/` - parameter/FLOP counts
- `figures/` - generated paper figures
- `logs/` - training/eval stdout logs

## License

MIT - see [LICENSE](LICENSE).
