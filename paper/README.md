# Manuscript

- **`FAT_SNN.tex`** — the version that matches the code in `Fat/`.
  Compile with an IEEE conference class (`pdflatex FAT_SNN.tex`) if
  `IEEEtran.cls` is installed.
- **`legacy/Fault_Aware_Training_For_SNN.pdf`** — the original three-page
  draft. Its Table I (~82 % / +2 pp) used analog training inputs and
  per-batch test scores. Do not quote those numbers next to the new
  results.

After `bash Fat/run_paper_experiments.sh` finishes, copy the mean ± std
row from `Fat/outputs/paper/table1/table1.md` into Table I of the tex.
