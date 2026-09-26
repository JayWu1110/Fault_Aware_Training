"""Train an SNN on MNIST with normal back-propagation or Fault Aware Training.

Examples
--------
Baseline (normal BP)::

    python train.py --mode normal --seed 0 --exp-name normal_s0

FAT: faults injected in every training forward pass, re-drawn per batch,
faulty rows restored after each optimiser step (paper III.A + III.B)::

    python train.py --mode fat --seed 0 --fault-prob 0.25 --fault-model hts --exp-name fat_s0

Outputs go to ``outputs/<exp-name>/``: ``best.ckpt``, ``config.json``,
``history.csv`` and ``train.log``.
"""
import argparse
import csv
import logging
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from generation import funcFAT, modules
from generation.params import Param, param as DEFAULT
from network import get_model
from utils import (ensure_dir, fault_generator, get_dataloaders,
                   maybe_extract_data_zip, prepare_batch, save_json, set_seed)


# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> Param:
    d = DEFAULT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["normal", "fat"], default="fat",
                   help="normal = plain BP baseline, fat = fault aware training")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--epochs", type=int, default=d.n_epochs)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--num-steps", type=int, default=d.num_steps)
    p.add_argument("--beta", type=float, default=d.beta)
    p.add_argument("--threshold", type=float, default=d.threshold)
    p.add_argument("--layers", type=int, nargs="+", default=d.layers)
    p.add_argument("--fault-prob", type=float, default=d.fault_prob)
    p.add_argument("--fault-model", choices=funcFAT.FAULT_KINDS, default=d.fault_model)
    p.add_argument("--no-recovery", action="store_true", help="disable weight recovery (III.B)")
    p.add_argument("--no-resample", action="store_true", help="draw the fault list once instead of per batch")
    p.add_argument("--include-output-faults", action="store_true",
                   help="also inject faults into the class layer (off by default)")
    p.add_argument("--fault-layers", type=int, nargs="+", default=None,
                   help="only inject faults into these layer ids (0-based)")
    p.add_argument("--select-ckpt", choices=["auto", "ff", "faulty"], default=d.select_ckpt,
                   help="which validation score writes best.ckpt and drives early stopping")
    p.add_argument("--dataset", choices=["mnist", "fashion"], default=d.dataset)
    p.add_argument("--arch", choices=["fc", "conv"], default=d.arch)
    p.add_argument("--device", default=d.device)
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--exp-name", default=None, help="defaults to <mode>_s<seed>")
    p.add_argument("--max-batches", type=int, default=0,
                   help="stop each epoch after this many batches (0 = full epoch; for smoke tests)")
    a = p.parse_args(argv)

    cfg = replace(
        d,
        mode=a.mode, seed=a.seed, n_epochs=a.epochs, patience=a.patience, batch_size=a.batch_size,
        lr=a.lr, weight_decay=a.weight_decay, grad_clip=a.grad_clip, num_steps=a.num_steps,
        beta=a.beta, threshold=a.threshold, layers=list(a.layers),
        fault_prob=a.fault_prob, fault_model=a.fault_model,
        weight_recovery=not a.no_recovery, resample_faults=not a.no_resample,
        exclude_output=not a.include_output_faults, fault_layers=a.fault_layers,
        select_ckpt=a.select_ckpt, dataset=a.dataset, arch=a.arch,
        device=a.device, data_root=a.data_root, out_dir=a.out_dir,
        exp_name=a.exp_name or f"{a.mode}_s{a.seed}",
        max_batches=a.max_batches,
    )
    return cfg


def setup_logger(out: Path) -> logging.Logger:
    log = logging.getLogger(f"fat.{out.name}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    return log


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_loader(model, loader, cfg: Param, faults_gen=None, fault_prob=None):
    """Return (loss, acc) over ``loader``.

    With ``faults_gen`` a fresh fault list is drawn per batch and injected.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    tot_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        x = prepare_batch(imgs, cfg)
        y = labels.to(cfg.device)
        faults = None
        if faults_gen is not None:
            tmp = replace(cfg, fault_prob=fault_prob if fault_prob is not None else cfg.fault_prob)
            faults = funcFAT.gen_faultlist_from_cfg(model, tmp, generator=faults_gen)
        logits = modules.forward_steps(model, x, cfg.num_steps, faults)
        tot_loss += criterion(logits, y).item()
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.numel()
    return tot_loss / n, correct / n


def train(cfg: Param) -> Path:
    mode = cfg.mode
    out = ensure_dir(Path(cfg.out_dir) / cfg.exp_name)
    log = setup_logger(out)
    set_seed(cfg.seed)
    maybe_extract_data_zip(cfg.data_root)

    cfg_dict = cfg.as_dict()
    save_json(cfg_dict, out / "config.json")
    log.info("config: %s", cfg_dict)

    train_loader, valid_loader, _ = get_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.valid_ratio, cfg.seed, dataset=cfg.dataset)
    model = get_model(cfg.model, cfg.layers, cfg.device,
                      beta=cfg.beta, threshold=cfg.threshold, arch=cfg.arch)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    g_train = fault_generator(cfg.seed)
    g_valid = fault_generator(cfg.seed + 1)

    use_faults = mode == "fat"
    fixed_faults = None
    if use_faults and not cfg.resample_faults:
        fixed_faults = funcFAT.gen_faultlist_from_cfg(model, cfg, generator=g_train)

    hist_path = out / "history.csv"
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc",
                                "valid_acc_faulty", "lr", "time_s"])

    best_ff, best_faulty, stale = -1.0, -1.0, 0
    kind = cfg.chosen_ckpt_kind()
    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, run_correct, seen = 0.0, 0, 0

        for b_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{cfg.n_epochs}", leave=False)):
            if cfg.max_batches and b_idx >= cfg.max_batches:
                break
            x = prepare_batch(imgs, cfg)
            y = labels.to(cfg.device)

            faults = None
            if use_faults:
                faults = fixed_faults if fixed_faults is not None else funcFAT.gen_faultlist_from_cfg(
                    model, cfg, generator=g_train)

            # III.B weight recovery: remember the rows we will restore after the step.
            saved_rows = None
            if use_faults and cfg.weight_recovery and faults:
                saved_rows = {(l, o): model.layers[l].weight.data[o].clone()
                              for (l, o) in funcFAT.faulty_rows(faults)}

            # Keep stuck-at weights in place through backward + step so autograd
            # sees the same values used in the forward pass.
            with modules.inject_weight_faults(model, faults or []):
                logits = modules.forward_steps(model, x, cfg.num_steps, faults, apply_swf=False)
                loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()

            if saved_rows:
                with torch.no_grad():
                    for (l, o), w in saved_rows.items():
                        model.layers[l].weight.data[o] = w

            run_loss += loss.item() * y.numel()
            run_correct += (logits.argmax(-1) == y).sum().item()
            seen += y.numel()

        train_loss, train_acc = run_loss / seen, run_correct / seen
        valid_loss, valid_acc = evaluate_loader(model, valid_loader, cfg)
        _, valid_acc_faulty = evaluate_loader(model, valid_loader, cfg, faults_gen=g_valid)
        dt = time.time() - t0

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{valid_loss:.5f}",
                                    f"{valid_acc:.5f}", f"{valid_acc_faulty:.5f}",
                                    optimizer.param_groups[0]["lr"], f"{dt:.1f}"])

        improved_ff = valid_acc > best_ff
        improved_ft = valid_acc_faulty > best_faulty
        selected = valid_acc_faulty if kind == "faulty" else valid_acc
        selected_best = best_faulty if kind == "faulty" else best_ff
        improved = selected > selected_best
        log.info("[%03d/%03d] train loss %.4f acc %.4f | valid loss %.4f acc %.4f faulty %.4f | %.0fs%s",
                 epoch, cfg.n_epochs, train_loss, train_acc, valid_loss, valid_acc, valid_acc_faulty, dt,
                 f" -> best ({kind})" if improved else "")

        if improved_ff:
            best_ff = valid_acc
            torch.save(model.state_dict(), out / "best_ff.ckpt")
        if improved_ft:
            best_faulty = valid_acc_faulty
            torch.save(model.state_dict(), out / "best_faulty.ckpt")
        if improved:
            stale = 0
            torch.save(model.state_dict(), out / "best.ckpt")
        else:
            stale += 1
            if stale > cfg.patience:
                log.info("no improvement on %s valid acc for %d epochs, early stopping", kind, cfg.patience)
                break

    torch.save(model.state_dict(), out / "last.ckpt")
    log.info("best valid ff %.4f / faulty %.4f (selected %s), checkpoint %s",
             best_ff, best_faulty, kind, out / "best.ckpt")
    return out


if __name__ == "__main__":
    train(parse_args())
