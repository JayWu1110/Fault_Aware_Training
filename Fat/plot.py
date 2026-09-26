"""Plotting helpers. All functions save PNGs and never call ``plt.show()``."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_history(csv_path, out_dir) -> Path:
    """Train / valid loss and accuracy per epoch."""
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(df.epoch, df.train_loss, label="train")
    ax1.plot(df.epoch, df.valid_loss, label="valid")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy"); ax1.set_title("loss"); ax1.legend()
    ax2.plot(df.epoch, df.train_acc, label="train")
    ax2.plot(df.epoch, df.valid_acc, label="valid (fault-free)")
    ax2.plot(df.epoch, df.valid_acc_faulty, label="valid (faulty)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_title("accuracy"); ax2.legend()
    fig.tight_layout()
    out = Path(out_dir) / "training_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_eval(csv_path, out_dir) -> list:
    """Box plot fault-free vs faulty and accuracy-vs-fault-rate curve."""
    df = pd.read_csv(csv_path)
    outs = []

    # --- box plot: fault-free vs each (fault model, p) -----------------
    groups, labels = [], []
    for (fm, p), g in df.groupby(["fault_model", "prob"], sort=False):
        groups.append(g.acc.values * 100)
        labels.append("fault-free" if fm == "none" else f"{fm}\np={p:g}")
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(groups)), 4))
    ax.boxplot(groups, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("fault-free vs. faulty accuracy (whole test set, repeated fault draws)")
    fig.tight_layout()
    out = Path(out_dir) / "ff_vs_faulty.png"
    fig.savefig(out, dpi=120); plt.close(fig); outs.append(out)

    # --- sweep: accuracy vs fault probability, one line per fault model
    faulty = df[df.fault_model != "none"]
    if faulty.prob.nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ff = df[df.fault_model == "none"].acc
        ax.axhline(ff.mean() * 100, color="k", ls="--", label="fault-free")
        for fm, g in faulty.groupby("fault_model"):
            s = g.groupby("prob").acc.agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"] * 100, yerr=s["std"] * 100, marker="o", capsize=3, label=fm)
        ax.set_xlabel("fault probability p"); ax.set_ylabel("test accuracy (%)")
        ax.set_title("accuracy vs. fault rate"); ax.legend()
        fig.tight_layout()
        out = Path(out_dir) / "acc_vs_fault_rate.png"
        fig.savefig(out, dpi=120); plt.close(fig); outs.append(out)
    return outs


def plot_table1(csv_path, out_dir) -> Path:
    """Paper Fig. 1: per-seed accuracy of normal vs modified BP."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, style in [("normal_ff", "b--o"), ("normal_faulty", "b-o"),
                       ("fat_ff", "r--s"), ("fat_faulty", "r-s")]:
        if col in df:
            ax.plot(df.seed, df[col] * 100, style, label=col.replace("_ff", " fault-free").replace("_faulty", " faulty"))
    ax.set_xlabel("experiment (seed)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("normal BP vs. FAT (modified BP)"); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = Path(out_dir) / "table1.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out
