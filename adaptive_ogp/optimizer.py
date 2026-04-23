"""Core optimizer: AdaptiveOGP (Algorithm 1 of the paper).

This optimizer wraps standard Adam with two design changes:

1. Moment-pathway routing. The first moment ``m_t`` is fed with the
   attenuated gradient ``tilde g_t = g_t - alpha_t U diag(sigma_hat) U^T g_t``;
   the second moment ``v_t`` is fed with the raw ``g_t^2``. See
   ``adaptive_ogp.routing`` for the 5-way routing enumeration and
   ``adaptive_ogp.subspace.SubspaceManager`` for the basis.

2. Overlap-aware adaptive strength. The scalar alpha_t is produced by
   ``adaptive_ogp.schedule.OverlapAwareSchedule`` from the per-step
   subspace-alignment signal ``s_t = ||U^T g_t||^2 / ||g_t||^2``.

The asymmetry in item 1 is the code-level content of Proposition 1
(the scalar surrogate): by preserving raw-gradient magnitude in the
denominator we avoid the ``attenuate-then-adapt conflict'' that
inflates the old-direction effective learning rate by 1/(1 - alpha).

Minimal usage::

    mgr = SubspaceManager(rank=32)
    opt = AdaptiveOGP(model.parameters(), lr=1e-4, subspace=mgr,
                      routing="ogp", schedule=OverlapAwareSchedule())
    # ... train task A, collect gradients into ``mgr`` via opt.collect(),
    # ... at task boundary: mgr.build_bases(); opt.on_task_switch().
    # thereafter, opt.step() applies Adaptive-OGP.

The optimizer supports ``routing in {vanilla, v_only, shared, ogp, reverse}``
so that Table 2 can be produced by a single codebase with a CLI flag.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import torch
from torch.optim import Optimizer

from adaptive_ogp.routing import (
    RoutingMode,
    apply_routing,
    compute_alignment,
    project_out,
)
from adaptive_ogp.schedule import OverlapAwareSchedule
from adaptive_ogp.subspace import SubspaceManager


class AdaptiveOGP(Optimizer):
    """Adam-compatible optimizer with decoupled moment routing.

    Parameters
    ----------
    params: iterable of torch.nn.Parameter
        Parameters to optimize.
    lr: float
        Base learning rate (Adam convention, before bias correction).
    betas: Tuple[float, float]
        (beta_1, beta_2) for Adam moments.
    eps: float
        Adam epsilon. The paper analysis assumes eps << sqrt(v_infty).
    weight_decay: float
        Decoupled weight-decay coefficient (AdamW-style).
    routing: str or RoutingMode
        One of {"vanilla", "v_only", "shared", "ogp", "reverse"}.
    subspace: SubspaceManager, optional
        Protected-subspace manager. Required for any routing other than
        "vanilla". The optimizer queries bases by parameter identity.
    schedule: OverlapAwareSchedule, optional
        Overlap-aware strength controller. If ``None``, ``alpha_t`` is
        held at ``alpha_max`` for every step (fixed-strength OGP).
    alpha_max: float
        Used only if ``schedule is None`` (fixed-strength mode).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        routing: Union[str, RoutingMode] = RoutingMode.OGP,
        subspace: Optional[SubspaceManager] = None,
        schedule: Optional[OverlapAwareSchedule] = None,
        alpha_max: float = 0.5,
    ) -> None:
        assert lr > 0.0, f"lr must be positive, got {lr}"
        assert 0.0 <= betas[0] < 1.0, f"beta_1 out of range: {betas[0]}"
        assert 0.0 <= betas[1] < 1.0, f"beta_2 out of range: {betas[1]}"
        assert eps > 0.0, f"eps must be positive, got {eps}"
        assert weight_decay >= 0.0, f"weight_decay negative: {weight_decay}"
        assert 0.0 <= alpha_max <= 1.0, f"alpha_max out of range: {alpha_max}"
        if isinstance(routing, str):
            routing = RoutingMode(routing)
        if routing != RoutingMode.VANILLA:
            assert subspace is not None, (
                "routing != VANILLA requires a SubspaceManager; pass one via `subspace=`"
            )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.routing: RoutingMode = routing
        self.subspace = subspace
        self.schedule = schedule
        self.alpha_max = alpha_max
        self._last_alpha: float = 0.0
        self._last_bar_s: float = 0.0

    # -- user-facing task-boundary hooks ------------------------------

    def collect(self, param: torch.Tensor, grad: torch.Tensor) -> None:
        """Forward a parameter/gradient pair into the subspace manager.

        Call this once per parameter per batch during the *old-task*
        collection window. At the task boundary, call ``on_task_switch``
        to trigger ``SubspaceManager.build_bases``.
        """
        assert self.subspace is not None, "collect() called without a SubspaceManager"
        self.subspace.collect(param, grad)

    def on_task_switch(self) -> None:
        """Materialize the SVD bases from buffered gradients."""
        assert self.subspace is not None, "on_task_switch() without SubspaceManager"
        self.subspace.build_bases()
        if self.schedule is not None:
            # Reset the EMA so that bar_s_t starts fresh on the new task.
            self.schedule.reset()

    # -- diagnostics ---------------------------------------------------

    @property
    def last_alpha(self) -> float:
        """Most recently used alpha_t. Useful for logging in experiments."""
        return self._last_alpha

    @property
    def last_bar_s(self) -> float:
        """Most recent smoothed overlap signal (``OverlapAwareSchedule.bar_s``)."""
        return self._last_bar_s

    # -- optimizer step ------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """One Adaptive-OGP update.

        The per-parameter body implements Algorithm 1 of the paper in
        PyTorch-native form, minus the closure pattern's loss value.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute a single global alpha_t (paper's canonical form); a
        # per-parameter alpha_t is a straightforward extension but is
        # intentionally out of scope for this reference implementation.
        alpha_t = self._compute_alpha()
        self._last_alpha = alpha_t

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert torch.isfinite(g).all(), (
                    f"NaN/Inf gradient for parameter shape {tuple(p.shape)}"
                )

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step_t = state["step"]
                m, v = state["m"], state["v"]

                # Build the attenuated gradient (tilde g) if a basis exists.
                basis, sigma_hat = (None, None)
                if self.subspace is not None:
                    basis, sigma_hat = self.subspace.basis_for(p)

                if basis is None:
                    g_mod = g  # no basis => pass through
                else:
                    basis_on_dev = basis.to(device=g.device, dtype=g.dtype)
                    sigma_on_dev = sigma_hat.to(device=g.device, dtype=g.dtype)
                    g_mod = project_out(g, basis_on_dev, sigma_on_dev, alpha_t)

                routed = apply_routing(raw_grad=g, modified_grad=g_mod, mode=self.routing)
                # Key asymmetry (Proposition 1): ``g_mod`` -> m; ``g`` -> v
                # for RoutingMode.OGP.
                m.mul_(beta1).add_(routed.num, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(routed.den, routed.den, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** step_t
                bias_corr2 = 1.0 - beta2 ** step_t
                denom = (v / bias_corr2).sqrt().add_(eps)
                step_size = lr / bias_corr1

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)

        return loss

    # -- internal ------------------------------------------------------

    def _compute_alpha(self) -> float:
        """Compute the per-step alpha_t from the overlap signal.

        Averages ``s_t`` across parameters that carry a protected basis.
        If no schedule is attached, returns ``alpha_max`` directly.
        """
        if self.routing == RoutingMode.VANILLA:
            return 0.0
        if self.subspace is None or not self.subspace.has_any_basis():
            return 0.0
        if self.schedule is None:
            self._last_bar_s = 0.0
            return self.alpha_max

        # aggregate alignment across all parameters that have a basis
        numer = 0.0
        count = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                basis, _ = self.subspace.basis_for(p)
                if basis is None:
                    continue
                s = compute_alignment(p.grad, basis.to(device=p.grad.device, dtype=p.grad.dtype))
                val = float(s.item()) if s.numel() == 1 else float(s.mean().item())
                assert math.isfinite(val), "non-finite subspace-alignment signal"
                numer += val
                count += 1
        s_mean = numer / max(count, 1)
        alpha_t = self.schedule.update(s_mean)
        self._last_bar_s = self.schedule.bar_s
        return alpha_t
