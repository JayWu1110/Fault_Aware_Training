"""Seeding, data loading and small IO helpers."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, random_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fault_generator(seed: int) -> torch.Generator:
    """Separate RNG stream for fault sampling (CPU generator)."""
    g = torch.Generator()
    g.manual_seed(seed + 10_007)
    return g


def _transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),          # [0, 1] -> used as spike probability
    ])


_DATASETS = {
    "mnist": torchvision.datasets.MNIST,
    "fashion": torchvision.datasets.FashionMNIST,
}


def get_dataloaders(data_root: str, batch_size: int, valid_ratio: float, seed: int,
                    download: bool = True, eval_batch_size: int = 512, dataset: str = "mnist"):
    """Train / valid / test loaders with an 8:2 split of the training set.

    Validation / test use a larger batch because no gradients are needed;
    the batch size does not change the accuracy numbers.
    """
    if dataset not in _DATASETS:
        raise ValueError(f"dataset must be one of {list(_DATASETS)}, got {dataset!r}")
    cls = _DATASETS[dataset]
    tfm = _transform()
    train_full = cls(root=data_root, train=True, download=download, transform=tfm)
    test_set = cls(root=data_root, train=False, download=download, transform=tfm)

    valid_size = int(valid_ratio * len(train_full))
    train_size = len(train_full) - valid_size
    train_set, valid_set = random_split(train_full, [train_size, valid_size],
                                        generator=torch.Generator().manual_seed(seed))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin)
    valid_loader = DataLoader(valid_set, batch_size=eval_batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, pin_memory=pin)
    return train_loader, valid_loader, test_loader


def flatten(imgs: torch.Tensor, device) -> torch.Tensor:
    return imgs.reshape(imgs.shape[0], -1).to(device, non_blocking=True)


def prepare_batch(imgs: torch.Tensor, cfg) -> torch.Tensor:
    """``(B, 784)`` for the FC net, ``(B, 1, 28, 28)`` for the conv net."""
    if getattr(cfg, "arch", "fc") == "conv":
        x = imgs.to(cfg.device, non_blocking=True)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x
    return flatten(imgs, cfg.device)


def resolve_ckpt(run_dir, kind: str = "auto", mode: str = "fat"):
    """Pick ``best.ckpt`` / ``best_faulty.ckpt`` / ``best_ff.ckpt`` from a run dir."""
    from pathlib import Path
    run_dir = Path(run_dir)
    if kind == "auto":
        kind = "faulty" if mode == "fat" else "ff"
    preferred = {
        "faulty": [run_dir / "best_faulty.ckpt", run_dir / "best.ckpt"],
        "ff": [run_dir / "best_ff.ckpt", run_dir / "best.ckpt"],
    }.get(kind, [run_dir / "best.ckpt"])
    for p in preferred:
        if p.exists():
            return p
    raise FileNotFoundError(f"no checkpoint in {run_dir} for kind={kind}")


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def maybe_extract_data_zip(data_root: str, zip_path: str = "data.zip") -> None:
    """Unpack the bundled ``data.zip`` if MNIST raw files are missing."""
    raw = Path(data_root) / "MNIST" / "raw"
    if raw.exists() and any(raw.iterdir()):
        return
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(Path(data_root).parent if Path(data_root).name == "data" else data_root)
