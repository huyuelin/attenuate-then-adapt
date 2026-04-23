"""Tests for the SVD basis extraction and groupwise SubspaceManager.

* Basis is (fan_in, r), orthonormal, and sigma_hat in [0,1].
* Non-matrix tensors are skipped (returned basis is None).
* Buffer capacity is respected.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from adaptive_ogp.subspace import SubspaceManager, extract_subspace_basis


def test_basis_shape_and_range():
    torch.manual_seed(0)
    G = torch.randn(64, 32)
    U, sigma = extract_subspace_basis(G, rank=8)
    assert U.shape == (32, 8)
    assert sigma.shape == (8,)
    assert torch.all(sigma >= 0.0)
    assert torch.all(sigma <= 1.0 + 1e-6)
    gram = U.T @ U
    assert torch.allclose(gram, torch.eye(8), atol=1e-4)


def test_extract_rejects_large_rank():
    G = torch.randn(10, 10)
    with pytest.raises(AssertionError):
        extract_subspace_basis(G, rank=15)


def test_manager_skips_1d():
    torch.manual_seed(1)
    mgr = SubspaceManager(rank=4, buffer_capacity=16)
    bias = torch.nn.Parameter(torch.zeros(8))
    for _ in range(20):
        mgr.collect(bias, torch.randn_like(bias))
    mgr.build_bases()
    assert not mgr.has_any_basis(), "1-D params should not receive a basis"


def test_manager_buffer_capacity():
    mgr = SubspaceManager(rank=4, buffer_capacity=16)
    W = torch.nn.Parameter(torch.randn(32, 8))
    for _ in range(100):
        mgr.collect(W, torch.randn_like(W))
    mgr.build_bases()
    assert mgr.has_any_basis()
    basis, _ = mgr.basis_for(W)
    assert basis is not None and basis.shape == (8, 4)
