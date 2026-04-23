"""Moment-pathway routing primitives.

Corresponds to Table 2 of the paper (``Moment-pathway 2x2 ablation``).
A ``routing'' decision is a pair ``(numerator_pathway, denominator_pathway)``
where each pathway independently selects either the attenuated gradient
``g_tilde = g - alpha U diag(sigma_hat) U^T g`` or the raw gradient ``g``.

The four modes studied in the paper are:

    vanilla     : (raw g -> m, raw g^2 -> v), no protection at all.
    v-only      : (raw g -> m, attenuated g^2 -> v); ablates the numerator.
    shared      : (attenuated g -> m, attenuated g^2 -> v); the classical
                  parameter-level routing used by OGD/GPM/Adam-NSCL/etc.
    ogp         : (attenuated g -> m, raw g^2 -> v); the Adaptive-OGP
                  asymmetry, and the routing that the paper advocates.
    reverse     : (raw g -> m, attenuated g^2 -> v); control used to rule
                  out ``any routing change helps''.

The asymmetry ``attenuated -> m, raw -> v`` is the single code-level
invariant the paper rests on; see Proposition 1 (the scalar surrogate)
and the causal isolation experiment (Table 3) for why the denominator
pathway is the load-bearing part.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch


class RoutingMode(str, Enum):
    """Enumeration of moment-pathway routings used in Table 2."""

    VANILLA = "vanilla"
    V_ONLY = "v_only"
    SHARED = "shared"
    OGP = "ogp"
    REVERSE = "reverse"


@dataclass(frozen=True)
class RoutedGradients:
    """Pair of (numerator_gradient, denominator_gradient).

    ``num`` is fed into the first-moment EMA ``m_t``; ``den`` is fed
    (squared) into the second-moment EMA ``v_t``.
    """

    num: torch.Tensor
    den: torch.Tensor


def apply_routing(
    raw_grad: torch.Tensor,
    modified_grad: torch.Tensor,
    mode: RoutingMode,
) -> RoutedGradients:
    """Dispatch the raw / modified gradient pair to the numerator and
    denominator pathways according to the requested routing mode.

    Parameters
    ----------
    raw_grad: torch.Tensor
        The unmodified gradient g_t.
    modified_grad: torch.Tensor
        The attenuated gradient tilde g_t = g_t - alpha_t U diag(sigma_hat) U^T g_t.
        For vanilla this should be passed equal to raw_grad.
    mode: RoutingMode
        One of RoutingMode.VANILLA, V_ONLY, SHARED, OGP, REVERSE.

    Returns
    -------
    RoutedGradients
        The (num, den) pair defining the moment-pathway routing.
    """
    assert raw_grad.shape == modified_grad.shape, (
        f"raw and modified gradient shape mismatch: "
        f"{tuple(raw_grad.shape)} vs {tuple(modified_grad.shape)}"
    )
    if mode == RoutingMode.VANILLA:
        return RoutedGradients(num=raw_grad, den=raw_grad)
    if mode == RoutingMode.V_ONLY:
        return RoutedGradients(num=raw_grad, den=modified_grad)
    if mode == RoutingMode.SHARED:
        return RoutedGradients(num=modified_grad, den=modified_grad)
    if mode == RoutingMode.OGP:
        return RoutedGradients(num=modified_grad, den=raw_grad)
    if mode == RoutingMode.REVERSE:
        return RoutedGradients(num=raw_grad, den=modified_grad)
    raise AssertionError(f"unknown routing mode {mode!r}")


def project_out(
    grad: torch.Tensor,
    basis: torch.Tensor,
    sigma_hat: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return ``g - alpha * U diag(sigma_hat) U^T g`` with shape-preserving
    handling for 1D and 2D parameter tensors.

    For non-matrix parameters (1D biases, layernorms) we fall back to an
    identity projection; the paper attaches SVD bases only to matrix-shaped
    groups (see ``SubspaceManager``).
    """
    assert 0.0 <= alpha <= 1.0, f"alpha out of range: {alpha}"
    if grad.ndim <= 1 or basis is None or basis.numel() == 0:
        return grad
    # Basis ``U`` is (fan_in, r); ``sigma_hat`` has shape (r,) in [0,1].
    # We act on the last dim of ``grad``.
    flat = grad.reshape(-1, grad.shape[-1])
    assert flat.shape[-1] == basis.shape[0], (
        f"basis/gradient last-dim mismatch: {flat.shape[-1]} vs {basis.shape[0]}"
    )
    coords = flat @ basis  # (N, r)
    scaled = coords * sigma_hat.unsqueeze(0)
    reconstruction = scaled @ basis.T  # (N, fan_in)
    modified = flat - alpha * reconstruction
    return modified.reshape_as(grad)


def compute_alignment(grad: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Scalar subspace-alignment signal s_t in [0,1].

    ``s_t = ||U^T g||^2 / ||g||^2`` with guards against zero gradients.
    The signal is the direct input to the overlap-aware schedule
    (``OverlapAwareSchedule``).
    """
    if grad.ndim <= 1 or basis is None or basis.numel() == 0:
        return torch.zeros((), device=grad.device, dtype=grad.dtype)
    flat = grad.reshape(-1, grad.shape[-1])
    coords = flat @ basis
    num = coords.pow(2).sum()
    den = flat.pow(2).sum().clamp_min(1.0e-20)
    ratio = (num / den).clamp(0.0, 1.0)
    return ratio


def shape_and_range_check(x: Tuple[torch.Tensor, ...]) -> None:
    """Fast-fail helper: no NaN, no Inf, finite dtype."""
    for i, t in enumerate(x):
        assert torch.isfinite(t).all(), f"tensor #{i} contains NaN/Inf"
