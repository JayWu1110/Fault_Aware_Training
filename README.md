# Fault Aware Training (FAT) for Spiking Neural Networks

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![snnTorch](https://img.shields.io/badge/snnTorch-0.9%2B-orange.svg)](https://github.com/jeshraghian/snntorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Train a spiking neural network so it keeps working when hidden neurons or synapses fail. Faults are injected **inside the training forward pass**; a fresh fault list is drawn every mini-batch, so healthy units learn to cover for broken ones.

Method inspired by Zahid et al., *FAT: Training Neural Networks for Reliable Inference Under Hardware Faults* (ITC 2020), applied to an SNN built with [snnTorch](https://github.com/jeshraghian/snntorch).

**Report:** [`report/report.pdf`](report/report.pdf)

---

## Table of contents

- [Features](#features)
- [Results](#results)
- [Method](#method)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Reproducing the report](#reproducing-the-report)
- [Default hyperparameters](#default-hyperparameters)
- [Outputs](#outputs)
- [Tests](#tests)
- [FAQ](#faq)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Features

- Fully connected LIF SNN (`784 → 256 → 32 → 10`) with a shared rate-coded path for train, validation, and test (`T = 25`)
- Three simulated hardware faults: hard-to-spike (`hts`), saturated (`sat`), stuck-at weight (`swf`)
- FAT training: inject faults in the forward pass, resample every batch, optional weight recovery
- Output (class) layer left healthy by default, so a dead logit cannot dominate the score
- Dual checkpoints: best fault-free and best faulty validation accuracy
- Official Table I (10 seeds, paired *t*-test), ablations, *p*-sweep, TMR vote, Fashion-MNIST
- Optional `--dataset fashion` and `--arch conv` (conv results are implemented, not in the report tables)

## Results

All numbers below are **hidden-layer** faults at \(p = 0.25\), \(R = 10\) full-test draws. They match [`report/report.pdf`](report/report.pdf). Do **not** quote the shipped `Fat/sample_best.ckpt` or the 2-epoch notes in `docs/results/` — those are not Table I.

### MNIST (10 seeds, hts)

| | Fault-free (%) | Faulty (%) |
|---|---:|---:|
| Normal BP | 97.72 ± 0.12 | 87.53 ± 2.48 |
| FAT | 97.90 ± 0.11 | **96.91 ± 0.21** |

Faulty-chip gain **+9.38 pp** (paired \(t = 12.18\), \(p = 6.8 \times 10^{-7}\)). Fault-free accuracy is essentially unchanged.

### Fashion-MNIST (3 seeds, hts)

| | Fault-free (%) | Faulty (%) |
|---|---:|---:|
| Normal BP | 87.78 ± 0.30 | 66.59 ± 1.39 |
| FAT | 86.32 ± 0.99 | **86.35 ± 0.46** |

Faulty-chip gain **+19.76 pp** (paired \(t = 34.7\), \(p = 8.3 \times 10^{-4}\)).

### What the ablations show

- **Resampling is the mechanism.** One frozen fault list for the whole run is *worse* than ordinary BP on hts (82.2% vs 86.9%).
- **Weight recovery is optional.** Turning it off leaves hts accuracy at 96.8% vs 97.0% for full FAT.
- **Faults do not transfer.** Train-on-hts / test-on-sat stays ~34%. Training under saturation is what repairs saturation (94.3%).
- **TMR** (3 independently faulted copies, majority vote) lifts normal BP to ~93.6% and still loses to a **single** FAT copy (~97.0%), at 3× inference cost.

## Method

Every MNIST / Fashion-MNIST pixel in \([0, 1]\) is a spike probability. At each of 25 time steps a Bernoulli spike is drawn and pushed through three `Linear → Leaky` stages (snnTorch LIF, \(\beta = 0.9\), threshold \(0.5\), subtract reset, arctan surrogate). Output spikes are summed and used as logits.

In `--mode fat`, each mini-batch:

1. Draws a new fault list (every **hidden** neuron faulty independently with probability \(p\)).
2. Runs the forward pass **with those faults injected**.
3. Clips the gradient \(\ell_2\) norm to 10 and takes an Adam step.
4. Optionally restores the incoming weight rows of units that were broken in that batch (`--no-recovery` skips this).

| Flag / name | Meaning |
|---|---|
| `hts` | Neuron never spikes (output forced to 0) |
| `sat` | Neuron spikes every step (output forced to 1) |
| `swf` | One random incoming synapse frozen in \([-10, 10]\) |
| `--no-resample` | One fault list for the whole run (ablation; hurts) |
| `--no-recovery` | Do not roll back faulty rows after Adam |
| `--include-output-faults` | Also break the class layer (off by default) |

`--mode normal` is the back-propagation baseline: no injection, no recovery.

Longer write-up: [`docs/METHOD.md`](docs/METHOD.md).

## Repository layout

```
Fat/
  train.py                 train one model (normal BP or FAT)
  evaluate.py              fault-free / faulty test accuracy + plots
  run_table1.py            official Table I over many seeds + paired t-test
  run_ablations.py         recovery / resample / cross-fault tests
  run_tmr.py               training-free triple vote baseline
  run_paper_experiments.sh overnight Table I then ablations
  run_paper_followup.sh    heatmap, p-sweep, TMR, Fashion-MNIST
  fat.py                   one-shot: train then evaluate
  plot.py                  figures (curves, heatmap, p-sweep)
  utils.py                 seeding, dataloaders, I/O
  network/SNN.py           784-256-32-10 LIF (optional conv)
  generation/
    params.py              default hyperparameters
    funcFAT.py             fault-list generation (hts / sat / swf)
    modules.py             shared forward pass
  tests/                   pytest sanity checks
  data.zip                 MNIST raw files (extracted on first run)
  sample_best.ckpt         LEGACY weights; not a FAT result
  requirements.txt
docs/
  METHOD.md                method as implemented
  results/                 historical / demo plots (not Table I)
report/report.pdf          manuscript
```

Checkpoints and logs go to `Fat/outputs/` (gitignored).

## Requirements

| | |
|---|---|
| OS | Linux (tested on Ubuntu / WSL2) |
| Python | 3.10 or newer (developed on 3.12) |
| GPU | CUDA optional; used automatically when `torch.cuda.is_available()` |
| Disk | MNIST in `Fat/data.zip`; Fashion-MNIST downloads via torchvision |

Python packages are pinned in [`Fat/requirements.txt`](Fat/requirements.txt): `torch`, `torchvision`, `snntorch`, `numpy`, `matplotlib`, `pandas`, `scipy`, `tqdm`, `pytest`.

## Installation

```bash
git clone https://github.com/JayWu1110/Fault_Aware_Training.git
cd Fault_Aware_Training/Fat

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

MNIST is extracted from `data.zip` to `Fat/data/` on the first training run. If the zip is missing, torchvision will download it.

## Quick start

All commands below are run from `Fat/` with the venv active.

```bash
# Baseline
python train.py --mode normal --seed 0

# FAT (hard-to-spike, 25% of hidden neurons, new list every batch)
python train.py --mode fat --seed 0 --fault-prob 0.25 --fault-model hts

# Evaluate the run
python evaluate.py --run outputs/fat_s0 --probs 0.05 0.1 0.25 0.5 \
    --fault-models hts sat swf --repeats 10
```

Runs write to `outputs/<mode>_s<seed>/` unless you pass `--exp-name`.

## Usage

### Training (`train.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `fat` | `normal` or `fat` |
| `--seed` | `777` | data / spike / fault RNG |
| `--epochs` | `20` | max epochs (early-stop patience 5) |
| `--batch-size` | `64` | |
| `--lr` | `3e-4` | Adam |
| `--fault-prob` | `0.25` | per-neuron fault probability |
| `--fault-model` | `hts` | `hts` / `sat` / `swf` |
| `--no-recovery` | off | disable weight rollback |
| `--no-resample` | off | freeze one fault list |
| `--dataset` | `mnist` | `mnist` or `fashion` |
| `--arch` | `fc` | `fc` or `conv` |
| `--select-ckpt` | `auto` | `auto` / `ff` / `faulty` |
| `--exp-name` | `<mode>_s<seed>` | output folder name |
| `--device` | auto | `cuda` or `cpu` |
| `--max-batches` | `0` | smoke-test cap (`0` = full epoch) |

```bash
python train.py --mode fat --dataset fashion --seed 0
python train.py --mode fat --arch conv --seed 0
python train.py --mode fat --no-recovery --seed 0 --exp-name hts_norecovery_s0
```

### Evaluation (`evaluate.py`)

```bash
python evaluate.py --run outputs/fat_s0 --probs 0.25 --fault-models hts --repeats 10

# Standalone checkpoint (not Table I)
python evaluate.py --ckpt sample_best.ckpt --out-dir outputs/legacy_sample \
    --probs 0.25 --fault-models hts --repeats 5
```

`--out-dir` controls where `eval.csv` is written, so a *p*-sweep does not overwrite Table I files.

### One-shot (`fat.py`)

```bash
python fat.py
```

Trains then evaluates with the defaults in `generation/params.py`. Prefer `train.py` / `evaluate.py` for anything you will cite.

## Reproducing the report

From `Fat/`. A full Table I (10 seeds × 20 epochs × two recipes) is an overnight GPU job.

```bash
# Table I — MNIST, 10 seeds, hidden-layer hts, p=0.25, R=10
python run_table1.py --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 20 --repeats 10 \
    --out-dir outputs/paper

# Ablations + cross-fault (3 seeds × 6 recipes)
python run_ablations.py --seeds 0 1 2 --epochs 20 --repeats 10

# Table I then ablations
bash run_paper_experiments.sh

# Heatmap, p-sweep, TMR, Fashion-MNIST Table I
bash run_paper_followup.sh
```

TMR on existing checkpoints:

```bash
python run_tmr.py --runs outputs/paper/normal_s0 outputs/paper/fat_s0 \
    --fault-prob 0.25 --repeats 10
```

## Default hyperparameters

| | Value |
|---|---|
| Architecture | FC `784-256-32-10`, no biases |
| LIF | \(\beta=0.9\), threshold \(0.5\), \(T=25\) |
| Optimiser | Adam, \(3\times 10^{-4}\), weight decay \(10^{-5}\) |
| Batch / split | 64, 8:2 train/val |
| Early stopping | patience 5 |
| Gradient clip | \(\ell_2\) norm 10 |
| Faults | hidden layers only, \(p=0.25\), resample on, recovery on |

## Outputs

Each training run writes `outputs/<exp-name>/`:

| File | Content |
|---|---|
| `config.json` | hyperparameters |
| `history.csv`, `train.log` | per-epoch loss / FF and faulty val accuracy |
| `best_ff.ckpt`, `best_faulty.ckpt`, `best.ckpt` | dual checkpoints; `best.ckpt` is the one used for test |
| `last.ckpt` | last epoch |
| `eval.csv`, `eval_summary.csv` | per-draw test accuracy and mean ± std |
| `training_curves.png`, `ff_vs_faulty.png`, `acc_vs_fault_rate.png` | figures |

`run_table1.py` also writes `table1.{csv,md,png}` under `--out-dir`.

`sample_best.ckpt` is from the original incomplete script (analog pixels at train time, no real forward-pass injection). Tests only use it to check state-dict keys. See [`Fat/LEGACY_CHECKPOINT.md`](Fat/LEGACY_CHECKPOINT.md).

## Tests

```bash
cd Fat
pytest -q
```

Checks include: checkpoint key match, fault-list rate near \(p\), hts zeros / sat ones, output layer excluded by default, recovery restores rows.

## FAQ

**Do I need a GPU?**  
No. CUDA is used when present. Table I on a laptop GPU (e.g. RTX 4060) is hours, not minutes; CPU is much slower.

**Why is the class layer not faulted?**  
At \(p=0.25\) a ten-way readout would lose two or three digits at random and dominate the error. Pass `--include-output-faults` if you want that protocol.

**Can I cite `docs/results/`?**  
No. That folder holds early / mixed-protocol runs. Cite `outputs/paper/table1/` after you reproduce, or the tables in [`report/report.pdf`](report/report.pdf).

**`git push` says fetch first.**  
The GitHub `README` was edited on the website after the first push, so histories diverged. Pull with rebase, or overwrite only if you intend to drop those remote commits.

## Limitations

- Faults are drawn independently per hidden neuron; real silicon defects are spatially correlated.
- No energy or accuracy numbers on a physical neuromorphic chip.
- TMR is a software majority vote, not chip-level lockstep.
- Convolutional SNN (`--arch conv`) is implemented but not in the report tables.
- We do not reimplement ReSpawn / hardware mappers; FAT is a training-loop change and is complementary to those.

## Roadmap

- [x] Shared rate-coded train/test path and real forward-pass injection
- [x] Table I, ablations, Fashion-MNIST, TMR, *p*-sweep
- [ ] Report convolutional SNN / N-MNIST numbers
- [ ] Measure on a neuromorphic chip
- [ ] Spatially correlated fault maps

## Contributing

Issues and pull requests are welcome.

1. Fork the repository
2. Create a branch (`git checkout -b feat/my-change`)
3. Run `pytest -q` from `Fat/`
4. Open a pull request against `master`

Please do not commit `Fat/outputs/`, `Fat/data/`, or venv files (already in `.gitignore`).

## Citation

If you use this code, please cite the report:

```bibtex
@techreport{wu2026fat,
  title  = {Fault Aware Training For Spiking Neural Networks},
  author = {Wu, Pin-Yu and Wang, Yu-Shiang},
  year   = {2026},
  institution = {National Taiwan University},
  url    = {https://github.com/JayWu1110/Fault_Aware_Training}
}
```

Related work this implementation follows:

```bibtex
@inproceedings{zahid2020fat,
  title     = {{FAT}: Training Neural Networks for Reliable Inference Under Hardware Faults},
  author    = {Zahid, Ussama and Gambardella, Giulio and Fraser, Nicholas J. and Blott, Michaela and Vissers, Kees},
  booktitle = {Proc. IEEE ITC},
  year      = {2020}
}
```

## Authors

- **Pin-Yu Wu** — [b11502041@ntu.edu.tw](mailto:b11502041@ntu.edu.tw)
- **Yu-Shiang Wang** — [b11504024@ntu.edu.tw](mailto:b11504024@ntu.edu.tw)

National Taiwan University, Taipei, Taiwan.

## Acknowledgments

- Zahid et al. (ITC 2020) for the FAT training recipe on non-spiking networks
- [snnTorch](https://github.com/jeshraghian/snntorch) (Eshraghian et al.) for LIF neurons and surrogate gradients
- Spyrou et al. (DATE 2021) and Putra et al. (ReSpawn, ICCAD 2021) for SNN fault-tolerance context

## License

This project is licensed under the [MIT License](LICENSE).
