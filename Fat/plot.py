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


def plot_ablations(csv_path, out_dir) -> Path:
    """Heatmap: training recipe (rows) vs test fault model (columns)."""
    df = pd.read_csv(csv_path)
    rows = ["normal", "hts", "hts_norecovery", "hts_noresample", "sat", "swf"]
    cols = [("test_hts", "hts"), ("test_sat", "sat"), ("test_swf", "swf")]
    labels = {
        "normal": "normal BP",
        "hts": "FAT (hts)",
        "hts_norecovery": "hts, no recovery",
        "hts_noresample": "hts, no resample",
        "sat": "FAT (sat)",
        "swf": "FAT (swf)",
    }
    present = [r for r in rows if r in set(df.train)]
    mat = []
    for r in present:
        g = df[df.train == r]
        mat.append([100 * g[c].mean() for c, _ in cols])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=25, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)), [n for _, n in cols])
    ax.set_yticks(range(len(present)), [labels.get(r, r) for r in present])
    ax.set_xlabel("test fault model")
    ax.set_title("cross-fault test accuracy (%)")
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v < 55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = Path(out_dir) / "ablations.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_psweep_compare(normal_csv, fat_csv, out_dir) -> Path:
    """Accuracy vs p for normal BP and FAT, one panel per fault model."""
    nd = pd.read_csv(normal_csv)
    fd = pd.read_csv(fat_csv)
    fms = [fm for fm in ("hts", "sat", "swf") if fm in set(nd.fault_model) | set(fd.fault_model)]
    fig, axes = plt.subplots(1, len(fms), figsize=(3.1 * len(fms), 3.2), sharey=True)
    if len(fms) == 1:
        axes = [axes]
    for ax, fm in zip(axes, fms):
        for df, name, style in ((nd, "normal BP", "o-"), (fd, "FAT", "s-")):
            g = df[df.fault_model == fm].groupby("prob").acc.agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"] * 100, yerr=g["std"] * 100,
                        fmt=style, capsize=3, label=name)
        ax.set_xlabel("fault probability $p$")
        ax.set_title(fm)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_dir) / "acc_vs_p.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
