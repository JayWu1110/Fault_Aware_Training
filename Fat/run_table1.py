"""Reproduce paper Table I: normal BP vs FAT over several seeds.

For every seed both models are trained (skipped when ``best.ckpt`` already
exists) and then evaluated fault-free and under ``--fault-prob`` with
``--repeats`` independent fault draws over the full test set.

Example (full paper setting)::

    python run_table1.py --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 20 --repeats 10

Outputs in ``outputs/table1/``: ``table1.csv`` (per seed), ``table1.md``
(mean +- std and a paired t-test) and ``table1.png``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import evaluate
import plot
import train
from utils import ensure_dir, get_dataloaders, maybe_extract_data_zip


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--fault-prob", type=float, default=0.25)
    p.add_argument("--fault-model", default="hts")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--quick", action="store_true",
                   help="short CPU demo: 2 seeds, 2 epochs, 3 fault draws (overridden by explicit flags)")
    p.add_argument("--force", action="store_true", help="retrain even if best.ckpt already exists")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="extra args forwarded to train.py (put after --extra)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.quick:
        a.seeds = a.seeds if a.seeds is not None else [0, 1]
        a.epochs = a.epochs if a.epochs is not None else 2
        a.repeats = a.repeats if a.repeats is not None else 3
    else:
        a.seeds = a.seeds if a.seeds is not None else list(range(10))
        a.epochs = a.epochs if a.epochs is not None else 20
        a.repeats = a.repeats if a.repeats is not None else 10
    table_dir = ensure_dir(Path(a.out_dir) / "table1")
    rows = []
    loader_cache = {}

    for seed in a.seeds:
        row = {"seed": seed}
        for mode in ("normal", "fat"):
            exp = f"{mode}_s{seed}"
            run_dir = Path(a.out_dir) / exp
            if a.force or not (run_dir / "best.ckpt").exists():
                cfg = train.parse_args(["--mode", mode, "--seed", str(seed), "--epochs", str(a.epochs),
                                        "--patience", str(a.patience), "--fault-prob", str(a.fault_prob),
                                        "--fault-model", a.fault_model, "--out-dir", a.out_dir,
                                        "--exp-name", exp] + a.extra)
                train.train(cfg)

            model, cfg = evaluate.load_run(run_dir)
            maybe_extract_data_zip(cfg.data_root)
            key = (cfg.data_root, cfg.batch_size, cfg.valid_ratio, cfg.seed, cfg.dataset)
            if key not in loader_cache:
                loader_cache[key] = get_dataloaders(
                    cfg.data_root, cfg.batch_size, cfg.valid_ratio, cfg.seed, dataset=cfg.dataset)[2]
            res = evaluate.run_eval(model, cfg, loader_cache[key], [a.fault_prob], [a.fault_model],
                                    a.repeats, seed=1000 + seed)
            evaluate.write_csv(res, run_dir / "eval.csv")
            df = pd.DataFrame(res)
            row[f"{mode}_ff"] = df[df.fault_model == "none"].acc.mean()
            row[f"{mode}_faulty"] = df[df.fault_model != "none"].acc.mean()
            row[f"{mode}_faulty_std"] = df[df.fault_model != "none"].acc.std(ddof=1) if a.repeats > 1 else 0.0
        rows.append(row)
        print(f"seed {seed}: " + " ".join(f"{k}={100*v:.2f}" for k, v in row.items() if k != "seed"))
        pd.DataFrame(rows).to_csv(table_dir / "table1.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(table_dir / "table1.csv", index=False)
    write_markdown(df, table_dir / "table1.md", a)
    plot.plot_table1(table_dir / "table1.csv", table_dir)
    print(open(table_dir / "table1.md").read())


def write_markdown(df: pd.DataFrame, path: Path, a) -> None:
    def ms(col):
        return f"{100*df[col].mean():.2f} ± {100*df[col].std(ddof=1) if len(df) > 1 else 0:.2f}"

    lines = [
        f"# Table I - normal BP vs FAT (fault model `{a.fault_model}`, p = {a.fault_prob}, "
        f"{a.repeats} fault draws per seed, {len(df)} seeds)",
        "",
        "| seed | normal fault-free | normal faulty | FAT fault-free | FAT faulty |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {int(r.seed)} | {100*r.normal_ff:.2f} | {100*r.normal_faulty:.2f} | "
                     f"{100*r.fat_ff:.2f} | {100*r.fat_faulty:.2f} |")
    lines.append(f"| **mean ± std** | {ms('normal_ff')} | {ms('normal_faulty')} | {ms('fat_ff')} | {ms('fat_faulty')} |")
    lines.append("")

    for label, ca, cb in [("fault-free", "normal_ff", "fat_ff"), ("faulty", "normal_faulty", "fat_faulty")]:
        diff = (df[cb] - df[ca]) * 100
        if len(df) > 1:
            t = stats.ttest_rel(df[cb], df[ca])
            lines.append(f"- {label}: FAT - normal = {diff.mean():+.2f} pp (std {diff.std(ddof=1):.2f}), "
                         f"paired t-test t = {t.statistic:.2f}, p = {t.pvalue:.3g}")
        else:
            lines.append(f"- {label}: FAT - normal = {diff.mean():+.2f} pp (single seed, no test)")
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
