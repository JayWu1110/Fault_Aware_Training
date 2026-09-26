"""Ablations and cross-fault evaluation.

Trains one FAT variant per row, then evaluates every model under hts / sat / swf
on the full test set (same p, R independent whole-chip draws).

Default variants
----------------
* ``hts``              recovery on, resample every batch (main recipe)
* ``hts_norecovery``   ``--no-recovery``
* ``hts_noresample``   ``--no-resample``
* ``sat``              train under saturated neurons
* ``swf``              train under stuck-at weights
* ``normal``           plain BP (no faults in training)

Example (paper setting)::

    python run_ablations.py --seeds 0 1 2 --epochs 20 --repeats 10

Short CPU check::

    python run_ablations.py --quick
"""
import argparse
from pathlib import Path

import pandas as pd

import evaluate
import train
from utils import ensure_dir, get_dataloaders, maybe_extract_data_zip

VARIANTS = (
    ("hts", ["--mode", "fat", "--fault-model", "hts"]),
    ("hts_norecovery", ["--mode", "fat", "--fault-model", "hts", "--no-recovery"]),
    ("hts_noresample", ["--mode", "fat", "--fault-model", "hts", "--no-resample"]),
    ("sat", ["--mode", "fat", "--fault-model", "sat"]),
    ("swf", ["--mode", "fat", "--fault-model", "swf"]),
    ("normal", ["--mode", "normal", "--fault-model", "hts"]),
)
TEST_MODELS = ("hts", "sat", "swf")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--fault-prob", type=float, default=0.25)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--out-dir", default="outputs/ablations")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--variants", nargs="+", default=None, help="subset of variant names")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.quick:
        a.seeds = a.seeds if a.seeds is not None else [0]
        a.epochs = a.epochs if a.epochs is not None else 3
        a.repeats = a.repeats if a.repeats is not None else 3
    else:
        a.seeds = a.seeds if a.seeds is not None else [0, 1, 2]
        a.epochs = a.epochs if a.epochs is not None else 20
        a.repeats = a.repeats if a.repeats is not None else 10
    wanted = set(a.variants) if a.variants else {n for n, _ in VARIANTS}
    variants = [(n, flags) for n, flags in VARIANTS if n in wanted]
    out = ensure_dir(Path(a.out_dir))
    rows = []
    loader_cache = {}

    for seed in a.seeds:
        for name, flags in variants:
            exp = f"{name}_s{seed}"
            run_dir = out / exp
            if a.force or not (run_dir / "best.ckpt").exists():
                argv = flags + ["--seed", str(seed), "--epochs", str(a.epochs),
                                "--patience", str(a.patience), "--fault-prob", str(a.fault_prob),
                                "--out-dir", str(out), "--exp-name", exp] + a.extra
                train.train(train.parse_args(argv))
            model, cfg = evaluate.load_run(run_dir)
            maybe_extract_data_zip(cfg.data_root)
            key = (cfg.data_root, cfg.batch_size, cfg.valid_ratio, cfg.seed, cfg.dataset)
            if key not in loader_cache:
                loader_cache[key] = get_dataloaders(
                    cfg.data_root, cfg.batch_size, cfg.valid_ratio, cfg.seed, dataset=cfg.dataset)[2]
            res = evaluate.run_eval(model, cfg, loader_cache[key], [a.fault_prob], list(TEST_MODELS),
                                    a.repeats, seed=2000 + seed)
            evaluate.write_csv(res, run_dir / "eval.csv")
            df = pd.DataFrame(res)
            row = {"seed": seed, "train": name}
            row["ff"] = df[df.fault_model == "none"].acc.mean()
            for fm in TEST_MODELS:
                sub = df[(df.fault_model == fm) & (df.prob == a.fault_prob)].acc
                row[f"test_{fm}"] = sub.mean()
                row[f"test_{fm}_std"] = sub.std(ddof=1) if len(sub) > 1 else 0.0
            rows.append(row)
            print(f"seed {seed} {name}: ff={100*row['ff']:.2f}  " +
                  " ".join(f"{fm}={100*row[f'test_{fm}']:.2f}" for fm in TEST_MODELS))
            pd.DataFrame(rows).to_csv(out / "ablations.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablations.csv", index=False)
    write_markdown(df, out / "ablations.md", a, TEST_MODELS)
    print(open(out / "ablations.md").read())


def write_markdown(df: pd.DataFrame, path: Path, a, test_models) -> None:
    lines = [
        f"# Ablations (p = {a.fault_prob}, {a.repeats} draws, {df.seed.nunique()} seed(s))",
        "",
        "| train | seed | fault-free | test hts | test sat | test swf |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.train} | {int(r.seed)} | {100*r.ff:.2f} | "
            f"{100*r.test_hts:.2f} | {100*r.test_sat:.2f} | {100*r.test_swf:.2f} |"
        )
    lines += ["", "## Mean over seeds", "",
              "| train | fault-free | test hts | test sat | test swf |",
              "|---|---:|---:|---:|---:|"]
    for name, g in df.groupby("train"):
        def ms(col):
            return f"{100*g[col].mean():.2f} ± {100*g[col].std(ddof=1) if len(g) > 1 else 0:.2f}"
        lines.append(f"| {name} | {ms('ff')} | {ms('test_hts')} | {ms('test_sat')} | {ms('test_swf')} |")
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
