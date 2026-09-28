"""Evaluate a trained checkpoint on the MNIST test set, fault-free and faulty.

For every (fault model, fault probability) pair ``--repeats`` independent
fault lists are drawn; each one is applied to the *whole* test set (one
simulated faulty chip) and the overall accuracy is recorded.  This replaces
the per-batch accuracies of the original ``fat.py`` whose 64-image batches
made the plots noisy.

Example::

    python evaluate.py --run outputs/fat_s0 --probs 0.05 0.1 0.25 0.5 \
        --fault-models hts sat swf --repeats 10

Writes ``eval.csv`` (raw), ``eval_summary.csv`` (mean/std) and plots into the
run directory.
"""
import argparse
import csv
from pathlib import Path
from typing import List

import torch

from generation import funcFAT, modules
from generation.params import Param
from network import get_model
from utils import (fault_generator, get_dataloaders, load_json, maybe_extract_data_zip,
                   prepare_batch, resolve_ckpt, set_seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="run directory containing best.ckpt and config.json")
    p.add_argument("--ckpt", default=None, help="checkpoint path (required if --run is omitted)")
    p.add_argument("--ckpt-kind", choices=["auto", "ff", "faulty"], default="auto",
                   help="which saved checkpoint to load from --run")
    p.add_argument("--out-dir", default=None, help="where to write eval csv/plots when --run is omitted")
    p.add_argument("--probs", type=float, nargs="+", default=[0.25])
    p.add_argument("--fault-models", nargs="+", choices=funcFAT.FAULT_KINDS, default=["hts"])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=12345, help="seed for fault sampling and spike encoding")
    p.add_argument("--device", default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def load_run(run_dir: Path = None, ckpt: str = None, device: str = None, ckpt_kind: str = "auto"):
    if run_dir and (Path(run_dir) / "config.json").exists():
        raw = load_json(Path(run_dir) / "config.json")
        known = {k: v for k, v in raw.items() if k in Param.__dataclass_fields__}
        cfg = Param(**known)
    else:
        cfg = Param()
    if device:
        cfg.device = device
    model = get_model(cfg.model, cfg.layers, cfg.device,
                      beta=cfg.beta, threshold=cfg.threshold, arch=cfg.arch)
    if ckpt:
        ckpt_path = Path(ckpt)
    elif run_dir:
        ckpt_path = resolve_ckpt(run_dir, kind=ckpt_kind, mode=cfg.mode)
    else:
        raise FileNotFoundError("need --run with a checkpoint or an explicit --ckpt")
    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def accuracy(model, loader, cfg: Param, faults=None) -> float:
    correct, n = 0, 0
    for imgs, labels in loader:
        x = prepare_batch(imgs, cfg)
        y = labels.to(cfg.device)
        logits = modules.forward_steps(model, x, cfg.num_steps, faults)
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.numel()
    return correct / n


def run_eval(model, cfg: Param, loader, probs: List[float], fault_models: List[str],
             repeats: int, seed: int) -> List[dict]:
    rows = []
    g = fault_generator(seed)
    for r in range(repeats):
        set_seed(seed + r)  # spike encoding noise differs per repeat
        rows.append(dict(fault_model="none", prob=0.0, repeat=r, acc=accuracy(model, loader, cfg)))
    for fm in fault_models:
        for p in probs:
            for r in range(repeats):
                set_seed(seed + r)
                faults = funcFAT.gen_faultlist(
                    model, p, fm, cfg.swf_low, cfg.swf_high, generator=g,
                    exclude_output=cfg.exclude_output, fault_layers=cfg.fault_layers)
                rows.append(dict(fault_model=fm, prob=p, repeat=r, n_faults=len(faults),
                                 acc=accuracy(model, loader, cfg, faults)))
    return rows


def summarise(rows: List[dict]) -> List[dict]:
    import numpy as np
    keys = sorted({(r["fault_model"], r["prob"]) for r in rows}, key=lambda k: (k[0] != "none", k))
    out = []
    for fm, p in keys:
        accs = np.array([r["acc"] for r in rows if r["fault_model"] == fm and r["prob"] == p])
        out.append(dict(fault_model=fm, prob=p, n=len(accs), mean=accs.mean(), std=accs.std(ddof=1) if len(accs) > 1 else 0.0,
                        min=accs.min(), max=accs.max()))
    return out


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1].keys()).index(k) if k in rows[-1] else 99)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    a = parse_args(argv)
    if not a.run and not a.ckpt:
        raise SystemExit("evaluate.py: provide --run DIR and/or --ckpt PATH")
    model, cfg = load_run(Path(a.run) if a.run else None, a.ckpt, a.device, a.ckpt_kind)
    if a.out_dir:
        out_dir = Path(a.out_dir)
    elif a.run:
        out_dir = Path(a.run)
    else:
        out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    maybe_extract_data_zip(cfg.data_root)
    _, _, test_loader = get_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.valid_ratio, cfg.seed, dataset=cfg.dataset)

    rows = run_eval(model, cfg, test_loader, a.probs, a.fault_models, a.repeats, a.seed)
    summary = summarise(rows)
    write_csv(rows, out_dir / "eval.csv")
    write_csv(summary, out_dir / "eval_summary.csv")

    print(f"{'fault':<6}{'p':>6}{'n':>4}{'mean%':>9}{'std%':>7}{'min%':>8}{'max%':>8}")
    for s in summary:
        print(f"{s['fault_model']:<6}{s['prob']:>6.2f}{s['n']:>4}{100*s['mean']:>9.2f}{100*s['std']:>7.2f}"
              f"{100*s['min']:>8.2f}{100*s['max']:>8.2f}")

    if not a.no_plots:
        import plot
        plot.plot_eval(out_dir / "eval.csv", out_dir)
        hist = (Path(a.run) / "history.csv") if a.run else (out_dir / "history.csv")
        if hist.exists():
            plot.plot_history(hist, out_dir)
    return summary


if __name__ == "__main__":
    main()
