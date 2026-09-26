"""Shared forward pass used by training and evaluation.

Training and inference must go through exactly the same code path
(rate-coded Bernoulli input, ``num_steps`` LIF steps, summed output spikes),
otherwise the network is trained on a different input distribution than the
one it is tested on.  ``forward_steps`` is that single path; ``good_inference``
and ``bad_inference`` are thin wrappers kept from the original project.
"""
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch

from .funcFAT import Fault, faults_to_masks


def apply_weight_faults(model, faults: List[Fault]) -> Dict[Tuple[int, int, int], torch.Tensor]:
    """Overwrite stuck-at synapses. Returns the original values for later restore."""
    saved: Dict[Tuple[int, int, int], torch.Tensor] = {}
    if not faults:
        return saved
    with torch.no_grad():
        for f in faults:
            if f.kind != "swf":
                continue
            w = model.layers[f.layer].weight
            key = (f.layer, f.out_idx, f.in_idx)
            if key not in saved:
                saved[key] = w[f.out_idx, f.in_idx].clone()
            w[f.out_idx, f.in_idx] = f.value
    return saved


def restore_weight_faults(model, saved: Dict[Tuple[int, int, int], torch.Tensor]) -> None:
    with torch.no_grad():
        for (layer, o, i), v in saved.items():
            model.layers[layer].weight[o, i] = v


@contextmanager
def inject_weight_faults(model, faults: List[Fault]):
    """Temporarily overwrite stuck-at synapses, restoring them on exit.

    Keep the context open through ``loss.backward()`` and ``optimizer.step()``
    so autograd sees the stuck values.  Evaluation can use the shorter
    ``forward_steps`` path, which restores immediately after the forward.
    """
    saved = apply_weight_faults(model, faults or [])
    try:
        yield
    finally:
        restore_weight_faults(model, saved)


def forward_steps(model, x: torch.Tensor, num_steps: int,
                  faults: Optional[List[Fault]] = None, encode: bool = True,
                  apply_swf: bool = True) -> torch.Tensor:
    """Run the SNN for ``num_steps`` steps, optionally under a fault list.

    Returns summed output spikes with shape ``(B, n_classes)``; treat them
    as logits for cross-entropy / argmax.

    ``apply_swf=False`` when the caller already applied stuck-at weights
    (training wraps the whole backward/step in ``inject_weight_faults``).
    """
    if not faults:
        return model(x, num_steps=num_steps, fault_masks=None, encode=encode)
    masks = faults_to_masks(model, faults, device=x.device)
    if apply_swf:
        with inject_weight_faults(model, faults):
            return model(x, num_steps=num_steps, fault_masks=masks, encode=encode)
    return model(x, num_steps=num_steps, fault_masks=masks, encode=encode)


def good_inference(model, x: torch.Tensor, num_steps: int = 25, encode: bool = True) -> torch.Tensor:
    """Fault-free inference."""
    return forward_steps(model, x, num_steps, faults=None, encode=encode)


def bad_inference(model, x: torch.Tensor, num_steps: int = 25,
                  fault_list: Optional[List[Fault]] = None, encode: bool = True) -> torch.Tensor:
    """Inference with ``fault_list`` injected."""
    return forward_steps(model, x, num_steps, faults=fault_list, encode=encode)
