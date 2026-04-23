"""Groupwise SVD basis extraction for the protected-direction manager.

Paper reference: Section 4 (``Routing'' paragraph) and Algorithm 1.
At task boundaries we buffer gradients, stack them into a tall matrix
``G`` of shape ``(num_samples, fan_in)`` per parameter group, and call
randomized SVD (Halko et al. 2011) to obtain a rank-``r`` basis ``U``
and normalized singular values ``sigma_hat = sigma / sigma.max()``.

For the reference implementation we use ``torch.svd_lowrank``, which is
randomized and deterministic given a fixed seed. The paper uses
``rank = 32`` by default for 256M HOPE; see configs/adaptive_ogp.yaml.

Non-matrix parameters (biases, layernorm scales) have no meaningful
low-rank direction and are therefore skipped (the manager returns
``None`` for them, which makes projections a no-op in ``routing.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import torch


def extract_subspace_basis(
    G: torch.Tensor,
    rank: int,
    n_oversample: int = 8,
    n_power_iterations: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomized SVD of a gradient buffer.

    Parameters
    ----------
    G: torch.Tensor, shape ``(num_samples, fan_in)``
        Row-stacked gradients collected on the old task(s).
    rank: int
        Target rank r; must be strictly less than ``min(G.shape)``.
    n_oversample: int
        Halko oversampling parameter for numerical stability.
    n_power_iterations: int
        Number of subspace-iteration sweeps for singular-value accuracy.

    Returns
    -------
    U: torch.Tensor, shape ``(fan_in, r)``
        Orthonormal basis of the top-r row-space of G.
    sigma_hat: torch.Tensor, shape ``(r,)``
        Normalized singular values, each in [0,1], sigma_hat[0] == 1.
    """
    assert G.ndim == 2, f"expected 2-D gradient buffer, got shape {tuple(G.shape)}"
    n, d = G.shape
    assert min(n, d) > rank, f"rank {rank} >= min(G.shape) = {min(n, d)}"
    q = min(rank + n_oversample, min(n, d) - 1)
    U_left, S, V = torch.svd_lowrank(G, q=q, niter=n_power_iterations)
    # V has shape (d, q); we want right singular vectors for row-space.
    basis = V[:, :rank].contiguous()
    sigma = S[:rank].contiguous()
    sigma_hat = sigma / sigma.max().clamp_min(1.0e-12)
    return basis, sigma_hat


@dataclass
class _GroupState:
    basis: Optional[torch.Tensor] = None
    sigma_hat: Optional[torch.Tensor] = None
    buffer: list = field(default_factory=list)


class SubspaceManager:
    """Per-group manager that owns the protected-subspace bases.

    Typical usage::

        mgr = SubspaceManager(rank=32)
        # while training task A:
        for p in model.parameters():
            if p.grad is None: continue
            mgr.collect(p, p.grad.detach())
        # at task boundary:
        mgr.build_bases()
        # thereafter, mgr.basis_for(p) returns (U, sigma_hat) or (None, None)

    The manager intentionally does not hold a reference to the optimizer;
    the optimizer queries the manager by parameter identity.
    """

    def __init__(
        self,
        rank: int = 32,
        buffer_capacity: int = 256,
        n_power_iterations: int = 2,
    ) -> None:
        assert rank > 0, f"rank must be positive, got {rank}"
        assert buffer_capacity >= rank + 4, (
            f"buffer_capacity={buffer_capacity} < rank+4={rank+4}"
        )
        self.rank = rank
        self.buffer_capacity = buffer_capacity
        self.n_power_iterations = n_power_iterations
        self._states: Dict[int, _GroupState] = {}

    def _key(self, param: torch.Tensor) -> int:
        return id(param)

    def collect(self, param: torch.Tensor, grad: torch.Tensor) -> None:
        """Push one flattened gradient row into the group's buffer.

        Non-matrix parameters are skipped (the manager owns only
        row-space bases for 2-D weight matrices).
        """
        assert torch.isfinite(grad).all(), "NaN/Inf in gradient during collection"
        if grad.ndim < 2:
            return
        key = self._key(param)
        state = self._states.setdefault(key, _GroupState())
        if len(state.buffer) >= self.buffer_capacity:
            return
        # For a weight matrix, rows (output units) are treated as samples;
        # the row-space (of dimension fan_in) is the protected subspace.
        flat = grad.detach().reshape(-1, grad.shape[-1])
        take = min(self.buffer_capacity - len(state.buffer), flat.shape[0])
        state.buffer.append(flat[:take].cpu())

    def build_bases(self) -> None:
        """Consume every buffered tensor and materialize (U, sigma_hat)."""
        for key, state in self._states.items():
            if not state.buffer:
                continue
            G = torch.cat(state.buffer, dim=0)
            if min(G.shape) <= self.rank:
                # fall back to no-projection when there is not enough
                # data to identify a rank-r subspace; fast-fail rather
                # than silently choose a wrong rank.
                state.basis = None
                state.sigma_hat = None
                state.buffer = []
                continue
            U, sigma_hat = extract_subspace_basis(
                G, rank=self.rank, n_power_iterations=self.n_power_iterations
            )
            state.basis = U
            state.sigma_hat = sigma_hat
            state.buffer = []  # release buffered memory

    def basis_for(
        self, param: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return ``(U, sigma_hat)`` for a parameter, or ``(None, None)``."""
        state = self._states.get(self._key(param))
        if state is None:
            return None, None
        return state.basis, state.sigma_hat

    def has_any_basis(self) -> bool:
        """True iff at least one group has a basis installed."""
        return any(s.basis is not None for s in self._states.values())

    def parameters_with_basis(self) -> Iterable[int]:
        """Ids of parameters that currently carry a protected basis."""
        for key, state in self._states.items():
            if state.basis is not None:
                yield key
