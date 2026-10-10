#!/usr/bin/env bash
#
# Full (non-smoke) DeepFakeBench training run for ForensicMTF.
#
# For each compression in COMPRESSIONS: builds the full FF++ index, trains
# ForensicMTF to convergence (early stopping), evaluates on the FF++ test
# split, then cross-domain-evaluates the same checkpoint on Celeb-DF-v1's official
# test split. Everything lands under records/DFB/.
#
# This is a multi-hour job on a single 11GB GPU (each compression is a full training
# run over ~1000 real + ~4000 fake FF++ videos). Runs one compression fully before
# starting the next - do not run this script twice concurrently, they'd contend for
# the same GPU.
#
# Usage:
#   ./scripts/run_dfb_full_training.sh                # trains c40 then c23 (default)
#   COMPRESSIONS="c40" ./scripts/run_dfb_full_training.sh   # c40 only
#   CONFIG=config_dfb.yaml ./scripts/run_dfb_full_training.sh
#
# Recommended: launch under nohup/tmux/screen since it will run for hours, e.g.:
#   nohup ./scripts/run_dfb_full_training.sh > records/DFB/logs/full_run_driver.log 2>&1 &

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config_dfb.yaml}"
COMPRESSIONS="${COMPRESSIONS:-c40 c23}"
LOG_DIR="records/DFB/logs"
mkdir -p "$LOG_DIR"

echo "=== ForensicMTF DFB full training run ==="
echo "Project root: $PROJECT_ROOT"
echo "Config:       $CONFIG"
echo "Compressions: $COMPRESSIONS"
echo "Python:       $(python3 --version) @ $(command -v python3)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "============================================"

for COMP in $COMPRESSIONS; do
    STAMP="$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo ">>> [$COMP] build_index  ($(date))"
    python3 main_dfb.py --config "$CONFIG" --compressions "$COMP" --task build_index \
        2>&1 | tee "$LOG_DIR/${COMP}_build_index_${STAMP}.log"

    echo ""
    echo ">>> [$COMP] train  ($(date)) - this is the long step, watch the log for progress"
    python3 main_dfb.py --config "$CONFIG" --compressions "$COMP" --task train \
        2>&1 | tee "$LOG_DIR/${COMP}_train_${STAMP}.log"

    echo ""
    echo ">>> [$COMP] evaluate (FF++ test split)  ($(date))"
    python3 main_dfb.py --config "$CONFIG" --compressions "$COMP" --task evaluate \
        2>&1 | tee "$LOG_DIR/${COMP}_evaluate_${STAMP}.log"

    echo ""
    echo ">>> [$COMP] cross_domain_eval (Celeb-DF-v1 test split)  ($(date))"
    python3 main_dfb.py --config "$CONFIG" --compressions "$COMP" --task cross_domain_eval \
        2>&1 | tee "$LOG_DIR/${COMP}_cross_domain_eval_${STAMP}.log"

    echo ""
    echo ">>> [$COMP] done  ($(date))"
    echo "    checkpoints: records/DFB/models_ffpp_${COMP}/best.pth"
    echo "    metrics csv: records/DFB/results/metrics_ffpp_${COMP}.csv"
    echo "    FF++ eval:   records/DFB/eval/dfb_ffpp_ffpp_${COMP}/metrics_summary.csv"
    echo "    CelebDF eval:records/DFB/eval/dfb_celebdf_ffpp_${COMP}/metrics_summary.csv"
done

echo ""
echo "=== All requested compressions complete ==="
