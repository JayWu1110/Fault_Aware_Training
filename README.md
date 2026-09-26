# Fault Aware Training (FAT) for Spiking Neural Networks

<<<<<<< HEAD
This project implements a **Fault Aware Training (FAT)** framework designed to improve the robustness and reliability of Spiking Neural Networks (SNNs) deployed on fault-prone neuromorphic hardware. 

## Overview
While SNNs are highly energy-efficient and biologically inspired, neuromorphic hardware is vulnerable to physical faults (e.g., neuron failures, synaptic errors, and random noise). Traditional error correction mechanisms add unwanted computational overhead. Our FAT framework solves this by integrating fault resilience directly into the software-level training process, teaching the network to adapt to errors dynamically.

## Key Techniques

Our modified backpropagation approach relies on three core mechanisms:

* **Fault Simulation:** During training, we simulate real-world hardware failures (e.g., "Hard to Spike" faults). We inject faults into neurons with a 25% probability, updating their weights with randomly selected wrong values (between -10 and 10).
* **Targeted Weight Recovery:** To prevent the network from completely overfitting to the simulated distortions, 25% of neurons are randomly selected during each iteration. Their weights are restored to their original values after backpropagation, stabilizing the learning process.
* **Gradient Clipping:** To stabilize deep SNN training and prevent exploding gradients caused by discrete spike activations, the maximum gradient norm is capped at a threshold of 10.

## Experimental Results
Evaluated on the **MNIST dataset**, the FAT framework consistently outperforms normal backpropagation. Key findings include:
* Higher classification accuracy in **both** fault-free and faulty hardware scenarios.
* An average accuracy improvement of **~2.0%** specifically under faulty conditions, proving the effectiveness of the targeted weight recovery and fault simulation.

## Environmental Setup
Download the tool for constructing virtual environment
```bash
sudo apt install python3-venv
```

Construct virtual environment
```bash
python3 -m venv venv
```

Enter into virtual environment
```bash
source venv/bin/activate
```

Install all required tool
```bash
pip3 install -r requirement.txt
```

Compilation and Executing
```bash
python3 fat.py
```



=======
Training a spiking neural network (SNN) on MNIST so that it keeps its accuracy
when neurons or synapses of the neuromorphic hardware are faulty. Faults are
simulated **inside the training forward pass**; the healthy part of the
network learns to compensate. Method inspired by Zahid et al., *FAT: Training
Neural Networks for Reliable Inference Under Hardware Faults* (ITC 2020),
applied to an SNN built with [snntorch](https://github.com/jeshraghian/snntorch).

Corrected manuscript (matches this code): [`paper/FAT_SNN.tex`](paper/FAT_SNN.tex).
The original draft PDF is archived at
[`paper/legacy/`](paper/legacy/) and must not be mixed with new numbers.
Method notes: [`docs/METHOD.md`](docs/METHOD.md).

## Layout

```
Fat/
  train.py          train one model (normal BP or FAT)
  evaluate.py       fault-free / faulty accuracy of a checkpoint, plots
  run_table1.py     reproduce Table I over many seeds (+ paired t-test)
  run_ablations.py  recovery / resample / hts-sat-swf cross tests
  run_paper_experiments.sh  overnight Table I + ablations
  fat.py            one-shot entry: train then evaluate
  sample_best.ckpt  LEGACY weights; see Fat/LEGACY_CHECKPOINT.md
  plot.py           figures
  utils.py          seeding, data loaders, IO helpers
  network/SNN.py    784-256-32-10 LIF network (snntorch Leaky)
  generation/
    params.py       default hyper-parameters
    funcFAT.py      fault list generation (hts / sat / swf)
    modules.py      shared forward pass, good_inference / bad_inference
  tests/            pytest sanity tests
  data.zip          MNIST raw files (extracted to Fat/data/ on first run)
  sample_best.ckpt  pre-trained weights from the original project
docs/legacy_results/  plots produced by the original fat.py
```

## Setup

Python ≥ 3.10. A CUDA GPU is used automatically when available.

```bash
cd Fat
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

MNIST is bundled in `Fat/data.zip`; it is extracted to `Fat/data/` the first
time you train (or torchvision downloads it if the zip is missing).

## Usage

All commands are run from `Fat/`.

```bash
# Normal back-propagation baseline
python train.py --mode normal --seed 0

# Fault Aware Training: hard-to-spike faults, 25 % of neurons, re-drawn every batch
python train.py --mode fat --seed 0 --fault-prob 0.25 --fault-model hts

# Evaluate a run: fault-free and under several fault rates / models
python evaluate.py --run outputs/fat_s0 --probs 0.05 0.1 0.25 0.5 \
    --fault-models hts sat swf --repeats 10

# Official Table I (hidden-layer faults only; writes outputs/paper/)
python run_table1.py --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 20 --repeats 10 \
    --out-dir outputs/paper

# Ablations + cross-fault tests
python run_ablations.py --seeds 0 1 2 --epochs 20 --repeats 10

# Overnight: Table I then ablations
bash run_paper_experiments.sh

# Optional extras
python train.py --mode fat --dataset fashion --seed 0
python train.py --mode fat --arch conv --seed 0

# Evaluate the shipped checkpoint without a training run
python evaluate.py --ckpt sample_best.ckpt --out-dir outputs/legacy_sample \
    --probs 0.25 --fault-models hts --repeats 5

# Tests
pytest -q
```

Useful `train.py` flags: `--epochs`, `--patience`, `--num-steps`, `--lr`,
`--no-recovery`, `--no-resample`, `--include-output-faults` (off by default),
`--select-ckpt {auto,ff,faulty}`, `--dataset {mnist,fashion}`, `--arch {fc,conv}`,
`--exp-name`.

`sample_best.ckpt` is from the original incomplete script — not a FAT result.

## Outputs

Each run writes to `outputs/<exp-name>/`:

| file | content |
|---|---|
| `config.json` | every hyper-parameter of the run |
| `history.csv`, `train.log` | per-epoch loss / accuracy (fault-free and faulty validation) |
| `best.ckpt`, `last.ckpt` | weights with best fault-free validation accuracy / final weights |
| `eval.csv`, `eval_summary.csv` | test accuracy per fault draw, and mean ± std |
| `training_curves.png`, `ff_vs_faulty.png`, `acc_vs_fault_rate.png` | figures |

`run_table1.py` additionally writes `outputs/table1/table1.{csv,md,png}`.
A copy of the 2-epoch CPU demonstration (Table I numbers and plots) is in
[`docs/results/`](docs/results/README.md).

## Method in one paragraph

Every pixel is a spike probability; a fresh Bernoulli spike train is drawn at
each of 25 time steps and pushed through three `Linear → Leaky` stages; output
spikes are summed and used as logits. In FAT mode each mini-batch draws a new
fault list (every neuron faulty with probability `p`), the forward pass is run
with those faults injected (hard-to-spike → output 0, saturated → output 1,
stuck-at weight → one synapse fixed to a value in [-10, 10]), gradients are
clipped to norm 10, Adam steps, and the incoming weights of the faulty neurons
are restored to their pre-step values. Evaluation applies whole-test-set fault
draws and reports mean ± std. See `docs/METHOD.md` for details and the list of
corrections the manuscript needs.
>>>>>>> f6ee51b (Rebuild FAT so training, evaluation, and the paper describe the same method.)
