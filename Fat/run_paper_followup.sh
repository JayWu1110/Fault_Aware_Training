#!/usr/bin/env bash
# Figures + optional extras after Table I / ablations.
# Safe to re-run: evaluate writes to outputs/psweep/, not over paper eval.csv.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-./venv/bin/python}"
LOG="${LOG:-outputs/paper_followup.log}"
mkdir -p outputs/psweep outputs/tmr outputs/fashion paper_figs
{
  echo "=== $(date) ablation heatmap ==="
  "$PYTHON" - <<'PY'
from pathlib import Path
import plot
src = Path("outputs/ablations/ablations.csv")
out = Path("outputs/ablations")
p = plot.plot_ablations(src, out)
print("wrote", p)
PY

  echo "=== $(date) p-sweep normal_s0 ==="
  "$PYTHON" -u evaluate.py --run outputs/paper/normal_s0 \
      --out-dir outputs/psweep/normal_s0 \
      --probs 0.05 0.1 0.25 0.5 --fault-models hts sat swf --repeats 10

  echo "=== $(date) p-sweep fat_s0 ==="
  "$PYTHON" -u evaluate.py --run outputs/paper/fat_s0 \
      --out-dir outputs/psweep/fat_s0 \
      --probs 0.05 0.1 0.25 0.5 --fault-models hts sat swf --repeats 10

  echo "=== $(date) compare p-sweep figure ==="
  "$PYTHON" - <<'PY'
from pathlib import Path
import plot
p = plot.plot_psweep_compare(
    "outputs/psweep/normal_s0/eval.csv",
    "outputs/psweep/fat_s0/eval.csv",
    "outputs/psweep")
print("wrote", p)
PY

  echo "=== $(date) TMR baseline ==="
  "$PYTHON" -u run_tmr.py \
      --runs outputs/paper/normal_s0 outputs/paper/fat_s0 \
             outputs/paper/normal_s1 outputs/paper/fat_s1 \
             outputs/paper/normal_s2 outputs/paper/fat_s2 \
      --fault-model hts --fault-prob 0.25 --repeats 10 \
      --out outputs/tmr/tmr.csv

  echo "=== $(date) Fashion-MNIST 3-seed Table I ==="
  "$PYTHON" -u run_table1.py --seeds 0 1 2 --epochs 20 --repeats 10 \
      --out-dir outputs/fashion --extra --dataset fashion

  echo "=== $(date) follow-up done ==="
} 2>&1 | tee -a "$LOG"
