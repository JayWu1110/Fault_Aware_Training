"""Fault list generation (paper section III.A).

Every neuron of every LIF layer is declared faulty with probability
``fault_prob``.  Three fault models are supported:

``hts``  hard-to-spike   - the neuron never emits a spike.
``sat``  saturated       - the neuron emits a spike at every time step.
``swf``  stuck-at weight - one randomly chosen incoming synapse of the
                           neuron is stuck at a random value drawn from
                           ``[swf_low, swf_high]`` (paper: -10 .. 10).

A fault is a :class:`Fault` record.  ``layer`` indexes ``model.layers`` /
``model.neurons`` (0-based), ``out_idx`` is the post-synaptic neuron in that
layer, ``in_idx`` (swf only) is the pre-synaptic index, i.e. the fault sits on
``model.layers[layer].weight[out_idx, in_idx]``.
"""
from typing import List, NamedTuple, Optional

import torch


class Fault(NamedTuple):
    kind: str            # "hts" | "sat" | "swf"
    layer: int
    out_idx: int
    in_idx: Optional[int] = None
    value: Optional[float] = None


FAULT_KINDS = ("hts", "sat", "swf")
_MASK_CODE = {"hts": 1, "sat": 2}


def layers_to_fault(model, exclude_output: bool = True,
                    fault_layers: Optional[List[int]] = None) -> List[int]:
    """Layer indices that may receive faults.

    The output (last) layer is skipped by default: silencing a class logit
    at ``p = 0.25`` makes whole-test-set accuracy jump by tens of points
    depending on which digits happen to die.
    """
    n = len(model.layers)
    if fault_layers is not None:
        return [i for i in fault_layers if 0 <= i < n]
    ids = list(range(n))
    if exclude_output and n:
        ids = ids[:-1]
    return ids


def gen_faultlist(model, fault_prob: float = 0.25, fault_model: str = "hts",
                  swf_low: float = -10.0, swf_high: float = 10.0,
                  generator: Optional[torch.Generator] = None,
                  exclude_output: bool = True,
                  fault_layers: Optional[List[int]] = None) -> List[Fault]:
    """Draw a fresh fault list for ``model``.

    ``generator`` lets callers make the draw reproducible independently of
    the global RNG used for data shuffling / spike encoding.
    """
    if fault_model not in FAULT_KINDS:
        raise ValueError(f"fault_model must be one of {FAULT_KINDS}, got {fault_model!r}")
    if not 0.0 <= fault_prob <= 1.0:
        raise ValueError("fault_prob must be in [0, 1]")

    faults: List[Fault] = []
    for layer_id in layers_to_fault(model, exclude_output, fault_layers):
        lin = model.layers[layer_id]
        n_out, n_in = lin.weight.shape[0], lin.weight.shape[1]
        hit = torch.rand(n_out, generator=generator) < fault_prob
        idx = torch.nonzero(hit, as_tuple=False).flatten().tolist()
        if fault_model == "swf" and idx:
            cols = torch.randint(0, n_in, (len(idx),), generator=generator).tolist()
            vals = (torch.rand(len(idx), generator=generator) * (swf_high - swf_low) + swf_low).tolist()
            faults.extend(Fault("swf", layer_id, o, c, v) for o, c, v in zip(idx, cols, vals))
        else:
            faults.extend(Fault(fault_model, layer_id, o) for o in idx)
    return faults


def gen_faultlist_from_cfg(model, cfg, generator: Optional[torch.Generator] = None) -> List[Fault]:
    """``gen_faultlist`` using the fields on a ``Param`` (or similar) object."""
    return gen_faultlist(
        model, getattr(cfg, "fault_prob", 0.25), getattr(cfg, "fault_model", "hts"),
        getattr(cfg, "swf_low", -10.0), getattr(cfg, "swf_high", 10.0),
        generator=generator,
        exclude_output=getattr(cfg, "exclude_output", True),
        fault_layers=getattr(cfg, "fault_layers", None),
    )


def faults_to_masks(model, faults: List[Fault], device=None) -> List[Optional[torch.Tensor]]:
    """Per-layer ``(n_out,)`` tensors with codes 0 / 1 (hts) / 2 (sat).

    Layers without neuron faults get ``None`` so the forward pass can skip
    the masking multiply.
    """
    device = device or next(model.parameters()).device
    masks: List[Optional[torch.Tensor]] = [None] * len(model.layers)
    for f in faults:
        if f.kind not in _MASK_CODE:
            continue
        if masks[f.layer] is None:
            masks[f.layer] = torch.zeros(model.layers[f.layer].weight.shape[0],
                                         dtype=torch.long, device=device)
        masks[f.layer][f.out_idx] = _MASK_CODE[f.kind]
    return masks


def faulty_rows(faults: List[Fault]):
    """Set of ``(layer, out_idx)`` pairs touched by any fault."""
    return {(f.layer, f.out_idx) for f in faults}
