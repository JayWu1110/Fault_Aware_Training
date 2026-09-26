"""Default hyper-parameters shared by training, evaluation and tests.

Everything here can be overridden from the command line (see ``train.py``
and ``evaluate.py``).  Keeping the defaults in one place means the paper's
Table I can be reproduced with ``python train.py --mode fat --seed 0``.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch


@dataclass
class Param:
    # ---- model ----
    model: str = "snn"
    arch: str = "fc"             # fc | conv
    # 784 -> 256 -> 32 -> 10 matches the shipped legacy ``sample_best.ckpt``.
    layers: List[int] = field(default_factory=lambda: [784, 256, 32, 10])
    beta: float = 0.9            # membrane decay of every LIF neuron
    threshold: float = 0.5       # firing threshold (per neuron, as in sample_best.ckpt)
    num_steps: int = 25          # simulation time steps per image
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data ----
    dataset: str = "mnist"       # mnist | fashion

    # ---- fault model ----
    fault_prob: float = 0.25     # probability that a hidden neuron is faulty
    fault_model: str = "hts"     # hts | sat | swf
    swf_low: float = -10.0       # stuck-at-weight range (paper III.A)
    swf_high: float = 10.0
    exclude_output: bool = True  # never inject faults into the class layer
    fault_layers: Optional[List[int]] = None  # if set, only these layer ids

    # ---- FAT training options ----
    mode: str = "fat"              # normal (plain BP baseline) | fat
    weight_recovery: bool = True   # paper III.B: restore faulty rows after step
    resample_faults: bool = True   # draw a fresh fault list every batch
    select_ckpt: str = "auto"      # auto | ff | faulty  (auto: fat->faulty, normal->ff)

    # ---- optimisation ----
    batch_size: int = 64
    n_epochs: int = 20
    patience: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 10.0
    valid_ratio: float = 0.2
    seed: int = 777

    # ---- io ----
    data_root: str = "data"
    out_dir: str = "outputs"
    exp_name: str = "sample"
    max_batches: int = 0           # 0 = full epoch; >0 limits batches (smoke tests)

    def as_dict(self):
        return asdict(self)

    def chosen_ckpt_kind(self) -> str:
        if self.select_ckpt in ("ff", "faulty"):
            return self.select_ckpt
        return "faulty" if self.mode == "fat" else "ff"


param = Param()
