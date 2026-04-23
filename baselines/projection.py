"""Shared-routing projection baselines.

The six methods covered here (OGD, GPM, SGP, ROGO, FOPNG, Adam-NSCL)
differ in how they build the protected subspace basis ``U`` and in the
choice of scaling applied to the coordinates. They are identical at the
moment-pathway level: all route the modified gradient into both the
first and the second Adam moments. That single commonality is the
attenuate-then-adapt conflict of Section 3 in the paper.

For the reference implementation we factor out the common routing
pathway into ``SharedRoutingProjection``; the six classes below are
thin subclasses that set the SVD recipe.

Paper references:
    OGD        : Farajtabar et al. (2020)
    GPM        : Saha et al. (2021)
    SGP        : Lee et al. (2022)
    ROGO       : Gong et al. (2022)
    FOPNG      : Guo et al. (2023)
    Adam-NSCL  : Wang et al. (2021)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim import Optimizer

from adaptive_ogp.routing import RoutingMode, apply_routing, project_out
from adaptive_ogp.subspace import SubspaceManager


@dataclass
class ProjectionConfig:
    """Hyperparameters that differentiate the six projection baselines."""

    name: str
    rank: int = 32
    alpha: float = 0.5
    use_sigma_scaling: bool = True  # sigma_hat vs. identity (1.0) on coordinates
    n_power_iterations: int = 2


class SharedRoutingProjection(Optimizer):
    """Adam + fixed-strength subspace projection with shared routing.

    This is the ``vanilla'' parameter-level routing studied in Table 2
    row ``shared''. Used as the canonical attenuate-then-adapt baseline.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        subspace: Optional[SubspaceManager] = None,
        config: Optional[ProjectionConfig] = None,
    ) -> None:
        assert lr > 0.0
        assert subspace is not None, "projection baseline requires a SubspaceManager"
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.subspace = subspace
        self.config = config or ProjectionConfig(name="shared")
        self._routing = RoutingMode.SHARED

    def collect(self, param: torch.Tensor, grad: torch.Tensor) -> None:
        self.subspace.collect(param, grad)

    def on_task_switch(self) -> None:
        self.subspace.build_bases()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        alpha = self.config.alpha if self.subspace.has_any_basis() else 0.0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert torch.isfinite(g).all()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]

                basis, sigma_hat = self.subspace.basis_for(p)
                if basis is None:
                    g_mod = g
                else:
                    U = basis.to(device=g.device, dtype=g.dtype)
                    if self.config.use_sigma_scaling:
                        sig = sigma_hat.to(device=g.device, dtype=g.dtype)
                    else:
                        sig = torch.ones_like(sigma_hat, device=g.device, dtype=g.dtype)
                    g_mod = project_out(g, U, sig, alpha)
                routed = apply_routing(g, g_mod, self._routing)
                m.mul_(beta1).add_(routed.num, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(routed.den, routed.den, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]
                denom = (v / bc2).sqrt().add_(eps)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


# -- Named subclasses (thin wrappers that only differ in config) -----

class OGD(SharedRoutingProjection):
    """Orthogonal Gradient Descent (Farajtabar et al., 2020).

    Uses an identity scaling on the coordinates (projection removes them
    in full); equivalent here to ``use_sigma_scaling=False``, alpha=1.0.
    """

    def __init__(self, params, subspace, **kw):
        cfg = ProjectionConfig(name="OGD", alpha=1.0, use_sigma_scaling=False)
        super().__init__(params, subspace=subspace, config=cfg, **kw)


class GPM(SharedRoutingProjection):
    """Gradient Projection Memory (Saha et al., 2021)."""

    def __init__(self, params, subspace, **kw):
        cfg = ProjectionConfig(name="GPM", alpha=1.0, use_sigma_scaling=False)
        super().__init__(params, subspace=subspace, config=cfg, **kw)


class SGP(SharedRoutingProjection):
    """Scaled Gradient Projection (Lee et al., 2022); uses sigma scaling."""

    def __init__(self, params, subspace, alpha: float = 0.5, **kw):
        cfg = ProjectionConfig(name="SGP", alpha=alpha, use_sigma_scaling=True)
        super().__init__(params, subspace=subspace, config=cfg, **kw)


class ROGO(SharedRoutingProjection):
    """Re-weighted Orthogonal Gradient Optimization (Gong et al., 2022)."""

    def __init__(self, params, subspace, alpha: float = 0.5, **kw):
        cfg = ProjectionConfig(name="ROGO", alpha=alpha, use_sigma_scaling=True)
        super().__init__(params, subspace=subspace, config=cfg, **kw)


class FOPNG(SharedRoutingProjection):
    """First-order Pseudo-null Gradient (Guo et al., 2023)."""

    def __init__(self, params, subspace, alpha: float = 0.5, **kw):
        cfg = ProjectionConfig(name="FOPNG", alpha=alpha, use_sigma_scaling=True)
        super().__init__(params, subspace=subspace, config=cfg, **kw)


class AdamNSCL(SharedRoutingProjection):
    """Adam with null-space continual learning projection (Wang et al., 2021).

    Same shared-routing pathway as above; the defining feature is the
    null-space (not row-space) basis extraction, which in this reference
    implementation is covered by the same SubspaceManager under a
    renaming of ``U <-> null(G)``.
    """

    def __init__(self, params, subspace, alpha: float = 0.5, **kw):
        cfg = ProjectionConfig(name="Adam-NSCL", alpha=alpha, use_sigma_scaling=True)
        super().__init__(params, subspace=subspace, config=cfg, **kw)
