# Recorded runs

## Official (hidden-layer faults only)

Produced by `bash Fat/run_paper_experiments.sh`.
Copy `Fat/outputs/paper/table1/table1.md` and
`Fat/outputs/ablations/ablations.md` here when that job finishes.

## Historical (2026-09-26, output layer also faulted)

`python run_table1.py --quick` — 2 seeds, 2 epochs, hts, $p=0.25$, 3 draws.
These numbers mixed dead class logits into the score and are **not** Table I.

| seed | normal fault-free | normal faulty | FAT fault-free | FAT faulty |
|---:|---:|---:|---:|---:|
| 0 | 96.75 | 64.34 | 95.27 | 69.09 |
| 1 | 96.04 | 63.05 | 95.11 | 78.12 |
| **mean ± std** | 96.40 ± 0.50 | 63.70 ± 0.91 | 95.19 ± 0.12 | 73.61 ± 6.39 |
