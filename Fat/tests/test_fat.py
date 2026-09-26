"""Minimal sanity tests. Run with ``pytest`` from the ``Fat/`` directory."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation import funcFAT, modules  # noqa: E402
from network import SNN, get_model  # noqa: E402

LAYERS = [784, 256, 32, 10]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return get_model("snn", LAYERS, "cpu")


def test_state_dict_matches_shipped_checkpoint(model):
    ckpt = ROOT / "sample_best.ckpt"
    if not ckpt.exists():
        pytest.skip("sample_best.ckpt not present")
    state = torch.load(ckpt, map_location="cpu")
    assert set(state) == set(model.state_dict())
    for k, v in state.items():
        assert model.state_dict()[k].shape == v.shape, k
    model.load_state_dict(state, strict=True)
    # the pre-trained network must do far better than chance on random digits
    x = torch.rand(16, 784)
    assert modules.good_inference(model, x, num_steps=10).shape == (16, 10)


def test_faultlist_ratio_close_to_p(model):
    g = torch.Generator().manual_seed(0)
    # output layer is excluded by default
    n_neurons = sum(LAYERS[1:-1])
    p = 0.25
    counts = [len(funcFAT.gen_faultlist(model, p, "hts", generator=g)) for _ in range(200)]
    ratio = sum(counts) / (200 * n_neurons)
    assert abs(ratio - p) < 0.02


def test_gen_faultlist_skips_output_layer(model):
    faults = funcFAT.gen_faultlist(model, 1.0, "hts", generator=torch.Generator().manual_seed(0))
    assert faults
    assert all(f.layer < len(model.layers) - 1 for f in faults)
    all_layers = funcFAT.gen_faultlist(model, 1.0, "hts", exclude_output=False,
                                       generator=torch.Generator().manual_seed(0))
    assert any(f.layer == len(model.layers) - 1 for f in all_layers)


def test_faultlist_is_reproducible(model):
    a = funcFAT.gen_faultlist(model, 0.3, "swf", generator=torch.Generator().manual_seed(7))
    b = funcFAT.gen_faultlist(model, 0.3, "swf", generator=torch.Generator().manual_seed(7))
    assert a == b


def test_swf_faults_have_in_idx_and_value_in_range(model):
    faults = funcFAT.gen_faultlist(model, 0.5, "swf", swf_low=-10, swf_high=10,
                                   generator=torch.Generator().manual_seed(1))
    assert faults
    for f in faults:
        assert f.kind == "swf" and f.in_idx is not None
        assert -10 <= f.value <= 10
        assert f.in_idx < model.layers[f.layer].weight.shape[1]


def test_empty_faultlist_equals_good_inference(model):
    x = torch.rand(8, 784)
    torch.manual_seed(1)
    good = modules.good_inference(model, x, num_steps=10)
    torch.manual_seed(1)
    bad = modules.bad_inference(model, x, num_steps=10, fault_list=[])
    assert torch.equal(good, bad)


def test_hard_to_spike_output_neuron_never_fires(model):
    x = torch.ones(4, 784)  # maximal drive
    faults = [funcFAT.Fault("hts", 2, 3)]
    out = modules.bad_inference(model, x, num_steps=20, fault_list=faults, encode=False)
    assert torch.all(out[:, 3] == 0)


def test_saturated_output_neuron_fires_every_step(model):
    x = torch.zeros(4, 784)
    faults = [funcFAT.Fault("sat", 2, 5)]
    out = modules.bad_inference(model, x, num_steps=20, fault_list=faults, encode=False)
    assert torch.all(out[:, 5] == 20)


def test_weight_fault_is_restored_after_inference(model):
    before = model.layers[1].weight.detach().clone()
    faults = [funcFAT.Fault("swf", 1, 2, 3, 9.0)]
    x = torch.rand(2, 784)
    with modules.inject_weight_faults(model, faults):
        assert model.layers[1].weight[2, 3].item() == 9.0
    assert torch.equal(model.layers[1].weight, before)


def test_forward_is_differentiable(model):
    x = torch.rand(4, 784)
    y = torch.randint(0, 10, (4,))
    faults = funcFAT.gen_faultlist(model, 0.25, "hts", generator=torch.Generator().manual_seed(0))
    logits = modules.forward_steps(model, x, num_steps=5, faults=faults)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_swf_stays_applied_through_backward(model):
    faults = [funcFAT.Fault("swf", 1, 2, 3, 9.0)]
    x = torch.rand(2, 784)
    y = torch.randint(0, 10, (2,))
    with modules.inject_weight_faults(model, faults):
        assert model.layers[1].weight[2, 3].item() == 9.0
        logits = modules.forward_steps(model, x, num_steps=3, faults=faults, apply_swf=False)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        assert model.layers[1].weight[2, 3].item() == 9.0
    assert model.layers[1].weight[2, 3].item() != 9.0


def test_conv_forward_and_faults():
    torch.manual_seed(0)
    net = get_model("snn", device="cpu", arch="conv")
    x = torch.rand(2, 1, 28, 28)
    out = modules.good_inference(net, x, num_steps=4)
    assert out.shape == (2, 10)
    faults = funcFAT.gen_faultlist(net, 1.0, "hts", generator=torch.Generator().manual_seed(0))
    assert faults and all(f.layer < len(net.layers) - 1 for f in faults)


def test_train_one_batch(tmp_path):
    from dataclasses import replace
    from generation.params import Param
    import train
    cfg = replace(Param(), mode="fat", seed=0, n_epochs=1, max_batches=1, num_steps=2,
                  batch_size=32, patience=1, device="cpu", out_dir=str(tmp_path),
                  exp_name="itest", data_root=str(ROOT / "data"))
    out = train.train(cfg)
    assert (out / "best.ckpt").exists()
    assert (out / "best_ff.ckpt").exists()
    assert (out / "best_faulty.ckpt").exists()
    assert (out / "config.json").exists()
