"""Fully connected spiking neural network built from snntorch LIF neurons.

The module layout (``layers.i`` / ``neurons.i``) is chosen so that the
state-dict keys match the shipped ``sample_best.ckpt``::

    layers.0.weight, layers.1.weight, layers.2.weight,
    neurons.i.{threshold, graded_spikes_factor, reset_mechanism_val, beta}
"""
from typing import List, Optional

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN(nn.Module):
    """``len(layers)-1`` linear layers, each followed by a Leaky neuron."""

    def __init__(self, layers: List[int], beta: float = 0.9, threshold: float = 0.5):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("need at least an input and an output size")
        self.sizes = list(layers)
        self.layers = nn.ModuleList(
            nn.Linear(i, o, bias=False) for i, o in zip(layers[:-1], layers[1:])
        )
        spike_grad = surrogate.atan()
        # One threshold value per neuron (shape ``(n,)``) - this is how the
        # shipped checkpoint stores it and it leaves room for per-neuron faults.
        self.neurons = nn.ModuleList(
            snn.Leaky(beta=beta, threshold=torch.full((n,), float(threshold)), spike_grad=spike_grad,
                      init_hidden=False, reset_mechanism="subtract")
            for n in layers[1:]
        )

    # ------------------------------------------------------------------ #
    def init_state(self, batch_size: int, device: torch.device):
        """Zero membrane potentials for every layer."""
        return [torch.zeros(batch_size, n, device=device) for n in self.sizes[1:]]

    def step(self, x: torch.Tensor, mems: List[torch.Tensor],
             fault_masks: Optional[List[torch.Tensor]] = None):
        """Advance the network by one time step.

        ``fault_masks`` (optional) is a list, one entry per layer, of tensors
        with shape ``(n_out,)`` holding:

        * ``0``  -> healthy neuron
        * ``1``  -> hard-to-spike (output forced to 0)
        * ``2``  -> saturated     (output forced to 1)

        Weight faults are applied by ``generation.modules`` directly on the
        parameters and need no mask.
        """
        spk = x
        new_mems = []
        for i, (lin, lif) in enumerate(zip(self.layers, self.neurons)):
            cur = lin(spk)
            spk, mem = lif(cur, mems[i])
            if fault_masks is not None and fault_masks[i] is not None:
                m = fault_masks[i]
                hts = (m == 1).to(spk.dtype)
                sat = (m == 2).to(spk.dtype)
                spk = spk * (1 - hts) * (1 - sat) + sat
            new_mems.append(mem)
        return spk, new_mems

    def forward(self, x: torch.Tensor, num_steps: int = 25,
                fault_masks: Optional[List[torch.Tensor]] = None,
                encode: bool = True) -> torch.Tensor:
        """Run ``num_steps`` steps and return summed output spikes.

        ``x`` holds pixel intensities in ``[0, 1]`` with shape ``(B, 784)``.
        With ``encode=True`` a fresh Bernoulli spike train is drawn at every
        step (rate coding); otherwise ``x`` is fed as a constant current.
        """
        mems = self.init_state(x.shape[0], x.device)
        out = torch.zeros(x.shape[0], self.sizes[-1], device=x.device)
        for _ in range(num_steps):
            inp = torch.bernoulli(x) if encode else x
            spk, mems = self.step(inp, mems, fault_masks)
            out = out + spk
        return out


class ConvSNN(nn.Module):
    """One 5×5 convolution + two linear LIF stages on 28×28 greyscale.

    Faults are injected only on ``self.layers`` (the fully-connected maps).
    The last linear layer is the class readout and is skipped by default
    when building a fault list.
    """

    def __init__(self, beta: float = 0.9, threshold: float = 0.5,
                 channels: int = 16, hidden: int = 128, n_classes: int = 10):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.sizes = [channels * 14 * 14, hidden, n_classes]
        self.conv = nn.Conv2d(1, channels, 5, padding=2, bias=False)
        self.pool = nn.AvgPool2d(2)
        spike_grad = surrogate.atan()
        self.conv_neuron = snn.Leaky(beta=beta, threshold=float(threshold), spike_grad=spike_grad,
                                     init_hidden=False, reset_mechanism="subtract")
        self.layers = nn.ModuleList([
            nn.Linear(self.sizes[0], hidden, bias=False),
            nn.Linear(hidden, n_classes, bias=False),
        ])
        self.neurons = nn.ModuleList(
            snn.Leaky(beta=beta, threshold=torch.full((n,), float(threshold)), spike_grad=spike_grad,
                      init_hidden=False, reset_mechanism="subtract")
            for n in (hidden, n_classes)
        )

    def init_state(self, batch_size: int, device: torch.device):
        mem_c = torch.zeros(batch_size, self.channels, 14, 14, device=device)
        mems = [torch.zeros(batch_size, n, device=device) for n in self.sizes[1:]]
        return mem_c, mems

    def step(self, x: torch.Tensor, mem_c: torch.Tensor, mems: List[torch.Tensor],
             fault_masks: Optional[List[torch.Tensor]] = None):
        cur = self.pool(self.conv(x))
        spk_c, mem_c = self.conv_neuron(cur, mem_c)
        spk = spk_c.flatten(1)
        new_mems = []
        for i, (lin, lif) in enumerate(zip(self.layers, self.neurons)):
            cur = lin(spk)
            spk, mem = lif(cur, mems[i])
            if fault_masks is not None and i < len(fault_masks) and fault_masks[i] is not None:
                m = fault_masks[i]
                hts = (m == 1).to(spk.dtype)
                sat = (m == 2).to(spk.dtype)
                spk = spk * (1 - hts) * (1 - sat) + sat
            new_mems.append(mem)
        return spk, mem_c, new_mems

    def forward(self, x: torch.Tensor, num_steps: int = 25,
                fault_masks: Optional[List[torch.Tensor]] = None,
                encode: bool = True) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        mem_c, mems = self.init_state(x.shape[0], x.device)
        out = torch.zeros(x.shape[0], self.sizes[-1], device=x.device)
        for _ in range(num_steps):
            inp = torch.bernoulli(x) if encode else x
            spk, mem_c, mems = self.step(inp, mem_c, mems, fault_masks)
            out = out + spk
        return out


def get_model(model: str = "snn", layers: Optional[List[int]] = None, device="cpu",
              beta: float = 0.9, threshold: float = 0.5, arch: str = "fc"):
    """Factory. ``arch='fc'`` is the paper Table I network; ``arch='conv'`` is extra."""
    if model.lower() != "snn":
        raise ValueError(f"unknown model type: {model}")
    if arch == "conv":
        net = ConvSNN(beta=beta, threshold=threshold)
    elif arch == "fc":
        net = SNN(layers or [784, 256, 32, 10], beta=beta, threshold=threshold)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return net.to(device)
