# FAT for SNN — method as implemented, and errata for the paper

This note pins down what the code in `Fat/` actually does.
`paper/FAT_SNN.tex` is the manuscript rewritten to match the code.
The original PDF is kept at `paper/Fault_Aware_Training_For_SNN.pdf`;
the "paper changes" below are the errata for that PDF.

## 1. Model

Fully connected SNN `784 → 256 → 32 → 10`, no biases, one snntorch `Leaky`
(LIF) neuron per unit, `beta = 0.9`, `threshold = 0.5`, subtract-reset,
arctan surrogate gradient. Defined in `network/SNN.py`. State-dict keys match
the shipped `Fat/sample_best.ckpt`.

## 2. Input encoding and time

Every MNIST pixel (in `[0, 1]`) is treated as a firing probability. At each of
`num_steps = 25` time steps a fresh Bernoulli spike train is drawn and fed to
the network. Output spikes of the last layer are summed over time and used as
logits for cross-entropy. **Training, validation and test all use this same
path** (`generation/modules.py::forward_steps`).

> Paper change: state explicitly that rate coding is used for training too.
> The original code trained on analog pixel values and only encoded spikes at
> test time; that mismatch is fixed.

## 3. Fault models (`generation/funcFAT.py`)

Every **hidden** neuron is faulty with probability `p` (default 0.25).
The output (class) layer is skipped unless you pass `--include-output-faults`.

| name | meaning | how it is simulated |
|---|---|---|
| `hts` | hard-to-spike | neuron output forced to 0 at every step |
| `sat` | saturated | neuron output forced to 1 at every step |
| `swf` | stuck-at weight | one random incoming synapse of the neuron is overwritten with a value drawn uniformly from `[-10, 10]` |

A fault list is a list of `Fault(kind, layer, out_idx, in_idx, value)` records.

> Paper change (III.A): the sentence "we let its weight update a wrong value
> randomly selected between -10 to 10" describes the `swf` model. Rename it
> and list the three models above. The default experiment uses `hts`.

## 4. Fault Aware Training (`train.py --mode fat`)

For every mini-batch:

1. Draw a fresh fault list (`--no-resample` keeps one list for the whole run).
2. **Fault simulation (III.A):** run the forward pass *with the faults
   injected*. The loss therefore reflects the faulty network, and back-prop
   teaches the healthy neurons to compensate.
3. Clip the gradient norm to 10 (III.C) and take an Adam step.
4. **Weight recovery (III.B):** restore the incoming weight rows of the faulty
   neurons to their pre-step values (`--no-recovery` disables this).

`--mode normal` skips 1, 2 and 4 and is the "Normal BP" baseline of Table I.

> Paper change (III.A/B): the original code did **not** inject faults in the
> forward pass; it only restored the rows of the "faulty" neurons after the
> optimiser step, so the loss never saw a fault. The text now matches the
> code: faults are injected during the forward pass, and weight recovery is a
> separate, optional stabiliser.

> Paper change (III.A): faults are re-drawn every batch (as in Zahid et al.),
> not fixed once per training run.

## 5. Evaluation (`evaluate.py`)

Each run writes `best_ff.ckpt` and `best_faulty.ckpt`. FAT early-stops on
faulty validation accuracy (`--select-ckpt auto`); normal BP uses the
fault-free score. The selected file is also saved as `best.ckpt`.
Evaluation uses the full 10 000-image test set:

* fault-free, repeated `R` times (encoding noise only);
* for each fault model and each `p`, `R` independent fault lists, each applied
  to the *whole* test set — one draw = one simulated faulty chip.

Reported numbers are mean ± std over the `R` draws. Plots: box plot
fault-free vs faulty, accuracy vs `p`, training curves.

> Paper change (III.D): replace the broken sentence
> "goodinf erence()f unctionf orf ault−f reescenariosandthe" with a
> description of `good_inference()` / `bad_inference()`, and replace the
> per-batch scatter/box plots (64 images per point) with whole-test-set
> statistics.

## 6. Table I (`run_table1.py`)

10 seeds × {normal, fat}, each evaluated fault-free and at `p = 0.25` with
10 fault draws. The script writes `outputs/table1/table1.md` with mean ± std
per column and a paired t-test on the per-seed differences.

> Paper change (IV, Table I): add mean ± std rows and the t-test p-value;
> "≈2 % improvement" must be accompanied by its uncertainty.

## 7. Reproducibility

`utils.set_seed` seeds `random`, `numpy`, `torch` (CPU and CUDA). Fault
sampling uses its own `torch.Generator` (`utils.fault_generator`) so changing
the data shuffling does not change the faults and vice-versa. Every run writes
its full configuration to `outputs/<exp>/config.json`.
