#!/usr/bin/env bash
# Overnight paper experiments. Run from Fat/:
#   bash run_paper_experiments.sh
# Resume is safe: existing best.ckpt directories are skipped unless you pass --force.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-./venv/bin/python}"
LOG="${LOG:-outputs/paper_experiments.log}"
mkdir -p outputs
echo "=== $(date) table I ===" | tee -a "$LOG"
"$PYTHON" -u run_table1.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --epochs 20 --repeats 10 --patience 5 \
    --out-dir outputs/paper \
    2>&1 | tee -a "$LOG"
echo "=== $(date) ablations ===" | tee -a "$LOG"
"$PYTHON" -u run_ablations.py \
    --seeds 0 1 2 --epochs 20 --repeats 10 \
    --out-dir outputs/ablations \
    2>&1 | tee -a "$LOG"
echo "=== $(date) done ===" | tee -a "$LOG"
