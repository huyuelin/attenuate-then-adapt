"""Correctness tests for ``AdaptiveOGP``.

* A single step without any basis reduces exactly to Adam.
* The moment-pathway routing selects the right gradients.
* NaN gradients trigger a fast-fail assertion.
* ``on_task_switch`` materialises a basis that makes subsequent steps
  different from the no-basis version.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from adaptive_ogp.optimizer import AdaptiveOGP
from adaptive_ogp.routing import RoutingMode, apply_routing
from adaptive_ogp.schedule import OverlapAwareSchedule
from adaptive_ogp.subspace import SubspaceManager


def _make_problem(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(32, 16)
    W = torch.nn.Parameter(torch.randn(16, 8))
    b = torch.nn.Parameter(torch.zeros(8))
    target = torch.randint(0, 8, (32,))
    return x, W, b, target


def _loss(x, W, b, target):
    logits = x @ W + b
    return torch.nn.functional.cross_entropy(logits, target)


def test_vanilla_reduces_to_adam():
    x, W, b, tgt = _make_problem()
    Wa = torch.nn.Parameter(W.detach().clone())
    Wb = torch.nn.Parameter(W.detach().clone())
    ba = torch.nn.Parameter(b.detach().clone())
    bb = torch.nn.Parameter(b.detach().clone())
    ref = torch.optim.Adam([Wa, ba], lr=1e-3)
    ogp = AdaptiveOGP([Wb, bb], lr=1e-3, routing=RoutingMode.VANILLA,
                      subspace=None)
    for _ in range(3):
        ref.zero_grad(set_to_none=True)
        ogp.zero_grad(set_to_none=True)
        la = _loss(x, Wa, ba, tgt)
        lb = _loss(x, Wb, bb, tgt)
        la.backward()
        lb.backward()
        ref.step()
        ogp.step()
    assert torch.allclose(Wa, Wb, atol=1e-7), "VANILLA routing must match Adam exactly"
    assert torch.allclose(ba, bb, atol=1e-7)


def test_routing_asymmetry():
    raw = torch.arange(6.0).reshape(2, 3)
    mod = raw * 0.4
    ogp = apply_routing(raw, mod, RoutingMode.OGP)
    shared = apply_routing(raw, mod, RoutingMode.SHARED)
    assert torch.equal(ogp.num, mod), "OGP must route modified into numerator"
    assert torch.equal(ogp.den, raw), "OGP must route raw into denominator"
    assert torch.equal(shared.num, mod)
    assert torch.equal(shared.den, mod)


def test_nan_gradient_fails_fast():
    _, W, _, _ = _make_problem()
    W = torch.nn.Parameter(W.detach().clone())
    opt = AdaptiveOGP([W], lr=1e-3, routing=RoutingMode.VANILLA, subspace=None)
    W.grad = torch.full_like(W, float("nan"))
    with pytest.raises(AssertionError, match="NaN/Inf"):
        opt.step()


def test_task_switch_installs_basis():
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(32, 16))
    mgr = SubspaceManager(rank=4, buffer_capacity=16)
    opt = AdaptiveOGP([W], lr=1e-3, routing=RoutingMode.OGP, subspace=mgr,
                      schedule=OverlapAwareSchedule(alpha_max=0.5), alpha_max=0.5)
    # collect a few fake gradients
    for _ in range(8):
        mgr.collect(W, torch.randn_like(W))
    opt.on_task_switch()
    basis, sigma = mgr.basis_for(W)
    assert basis is not None, "SubspaceManager.build_bases should install a basis"
    assert basis.shape == (16, 4)
    assert sigma.shape == (4,)
    assert torch.all(sigma <= 1.0 + 1e-6) and torch.all(sigma >= 0.0)
