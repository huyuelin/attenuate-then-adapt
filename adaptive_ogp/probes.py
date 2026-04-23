"""Diagnostic probes used to reproduce Figure 1 and Tables 3 to 4.

These probes read state out of an ``AdaptiveOGP`` optimizer (or any
Adam-compatible optimizer that stores ``state[p]['v']``) and compute the
per-direction quantities that are the empirical subject of the paper:

* ``VtEnergyProbe``: old-direction second-moment energy
      E_old(t) = Tr( U^T * diag(v_t) * U ) / Tr(diag(v_t)).
  Figure 1 (A): the paper reports ~3.83x steady-state ratio between
  shared and decoupled routings at alpha = 0.5.
* ``EtaEffProbe``: old-direction effective learning-rate ratio
      R_eta(t) = || U^T ( lr / (sqrt(v_t) + eps) ) ||_F / || lr / (sqrt(v_t^ref) + eps) ||_F
  Figure 1 (B): the paper reports ~2.14x steady-state ratio (predicted 2x).
* ``SubspaceAlignmentProbe``: raw ``s_t = ||U^T g_t||^2 / ||g_t||^2``.

These are pure readers; they do not mutate optimizer state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from adaptive_ogp.subspace import SubspaceManager


@dataclass
class VtEnergyProbe:
    """Old-direction second-moment energy E_old(t).

    Parameters
    ----------
    subspace: SubspaceManager
        Provides the protected basis U per parameter.

    Notes
    -----
    The probe records a scalar per call; its output is the list of
    ``E_old(t)`` values aggregated across parameters that carry a basis.
    """

    subspace: SubspaceManager
    history: List[float] = field(default_factory=list)

    def measure(self, optimizer) -> float:
        numer = 0.0
        denom = 0.0
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                v = state.get("v")
                if v is None or v.ndim < 2:
                    continue
                basis, _ = self.subspace.basis_for(p)
                if basis is None:
                    continue
                v_flat = v.reshape(-1, v.shape[-1])
                U = basis.to(device=v_flat.device, dtype=v_flat.dtype)
                # row-weighted: v enters as diagonal of a (fan_in,fan_in) matrix
                # per output row; we sum E_old across output rows.
                proj = (v_flat @ U).pow(2).sum()  # Frobenius(U^T diag(v) U) proxy
                total = v_flat.pow(2).sum().clamp_min(1.0e-20)
                numer += float((proj / total).item())
                denom += 1.0
        val = numer / max(denom, 1.0)
        self.history.append(val)
        return val


@dataclass
class EtaEffProbe:
    """Old-direction effective learning-rate proxy R_eta(t).

    We report a per-parameter average of
        || U^T (1 / (sqrt(v) + eps)) ||_F / || (1 / (sqrt(v) + eps)) ||_F
    which is monotone in the classical old-direction eta_eff ratio and
    is directly comparable between routings (it absorbs the global ``lr``).
    """

    subspace: SubspaceManager
    eps: float = 1.0e-8
    history: List[float] = field(default_factory=list)

    def measure(self, optimizer) -> float:
        numer = 0.0
        denom = 0.0
        for group in optimizer.param_groups:
            eps = group.get("eps", self.eps)
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                v = state.get("v")
                if v is None or v.ndim < 2:
                    continue
                basis, _ = self.subspace.basis_for(p)
                if basis is None:
                    continue
                inv_sqrt = 1.0 / (v.sqrt() + eps)
                flat = inv_sqrt.reshape(-1, inv_sqrt.shape[-1])
                U = basis.to(device=flat.device, dtype=flat.dtype)
                old = (flat @ U).pow(2).sum()
                tot = flat.pow(2).sum().clamp_min(1.0e-20)
                numer += float((old / tot).item())
                denom += 1.0
        val = numer / max(denom, 1.0)
        self.history.append(val)
        return val


@dataclass
class SubspaceAlignmentProbe:
    """Raw per-step alignment signal s_t = ||U^T g||^2 / ||g||^2."""

    subspace: SubspaceManager
    history: List[float] = field(default_factory=list)

    def measure(self, optimizer) -> float:
        numer = 0.0
        denom = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                basis, _ = self.subspace.basis_for(p)
                if basis is None or p.grad.ndim < 2:
                    continue
                flat = p.grad.reshape(-1, p.grad.shape[-1])
                U = basis.to(device=flat.device, dtype=flat.dtype)
                coords = flat @ U
                num = coords.pow(2).sum()
                den = flat.pow(2).sum().clamp_min(1.0e-20)
                numer += float((num / den).item())
                denom += 1
        val = numer / max(denom, 1)
        self.history.append(val)
        return val


def summarize_histories(
    probes: Dict[str, object],
) -> Dict[str, Optional[float]]:
    """Utility for JSON logging: last value of each probe's history."""
    out: Dict[str, Optional[float]] = {}
    for name, probe in probes.items():
        hist = getattr(probe, "history", None)
        out[name] = None if not hist else hist[-1]
    return out
