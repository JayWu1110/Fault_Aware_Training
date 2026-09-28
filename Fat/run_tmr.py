"""Training-free triple modular redundancy (TMR) baseline.

For each test image, three independently drawn fault lists are applied to
the same checkpoint and the class votes are majority-pooled.  This is a
classical hardware-redundancy baseline (3x inference cost), not a
reimplementation of ReSpawn or Spyrou et al.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

import evaluate
from generation import funcFAT, modules
from utils import fault_generator, get_dataloaders, maybe_extract_data_zip, prepare_batch, set_seed


def tmr_accuracy(model, loader, cfg, fault_lists) -> float:
    correct, n = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            x = prepare_batch(imgs, cfg)
            y = labels.to(cfg.device)
            votes = []
            for faults in fault_lists:
                pred = modules.forward_steps(model, x, cfg.num_steps, faults).argmax(-1)
                votes.append(pred)
            stacked = torch.stack(votes, dim=0)
            # majority over 3 integer class votes
            maj = stacked.mode(dim=0).values
            correct += (maj == y).sum().item()
            n += y.numel()
    return correct / n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--fault-prob", type=float, default=0.25)
    p.add_argument("--fault-model", default="hts")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--voters", type=int, default=3)
    p.add_argument("--out", default="outputs/tmr/tmr.csv")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    rows = []
    for run in a.runs:
        run_dir = Path(run)
        model, cfg = evaluate.load_run(run_dir)
        maybe_extract_data_zip(cfg.data_root)
        _, _, loader = get_dataloaders(
            cfg.data_root, cfg.batch_size, cfg.valid_ratio, cfg.seed, dataset=cfg.dataset)
        g = fault_generator(3000 + sum(ord(c) for c in run_dir.name))
        accs = []
        for r in range(a.repeats):
            set_seed(4000 + r)
            lists = [
                funcFAT.gen_faultlist(
                    model, a.fault_prob, a.fault_model, cfg.swf_low, cfg.swf_high,
                    generator=g, exclude_output=cfg.exclude_output,
                    fault_layers=cfg.fault_layers)
                for _ in range(a.voters)
            ]
            accs.append(tmr_accuracy(model, loader, cfg, lists))
        series = pd.Series(accs)
        rows.append({
            "run": run_dir.name, "fault_model": a.fault_model, "prob": a.fault_prob,
            "voters": a.voters, "mean": series.mean(), "std": series.std(ddof=1),
            "n": len(accs),
        })
        print(f"{run_dir.name}: TMR-{a.voters} {a.fault_model} p={a.fault_prob} "
              f"{100*series.mean():.2f} ± {100*series.std(ddof=1):.2f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
