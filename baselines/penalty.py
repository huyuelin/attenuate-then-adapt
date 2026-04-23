"""Penalty-based baselines: EWC and SI.

Both methods estimate an importance weight ``Omega`` per parameter and
multiply the gradient coordinate-wise by ``(1 - lambda * Omega)`` (after
clamping to [0,1]). In the paper this is the ``penalty-family'' row of
Table 7, and the ``shared'' baseline for the 2x2 penalty ablation in
Section ``Cross-family breadth''.

* ``EWC``: Omega := diagonal Fisher estimated on the old-task data.
* ``SI``:  Omega := Synaptic Intelligence path integral (Zenke et al., 2017).

The attenuate-then-adapt conflict in this family is identical in spirit
to the projection case: the rescaled gradient enters both Adam moments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from torch.optim import Optimizer

from adaptive_ogp.routing import RoutedGradients, RoutingMode


class SharedRoutingPenalty(Optimizer):
    """Adam with per-coordinate shared-routing penalty rescaling.

    Parameters
    ----------
    params, lr, betas, eps, weight_decay: Adam hyperparameters.
    lambda_penalty: scalar multiplying the importance map ``Omega``.
        With the convention ``(1 - lambda * Omega).clamp(0,1)`` applied
        to the gradient, a larger ``lambda`` produces stronger attenuation
        in important coordinates.

    Notes
    -----
    The importance map ``Omega`` is owned by the subclass and is refreshed
    at task boundaries via ``update_importance(new_importance_map)``.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        lambda_penalty: float = 0.5,
    ) -> None:
        assert lr > 0.0
        assert 0.0 <= lambda_penalty, f"lambda_penalty negative: {lambda_penalty}"
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        Optimizer.__init__(self, params, defaults)
        self.lambda_penalty = lambda_penalty
        self._omega: Dict[int, torch.Tensor] = {}
        self._routing = RoutingMode.SHARED

    def update_importance(self, importance: Dict[int, torch.Tensor]) -> None:
        """Refresh the importance map; keys are ``id(param)``."""
        for k, v in importance.items():
            assert torch.isfinite(v).all(), "NaN in importance"
        self._omega = importance

    def _attenuated(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        omega = self._omega.get(id(p))
        if omega is None:
            return g
        omega = omega.to(device=g.device, dtype=g.dtype)
        factor = (1.0 - self.lambda_penalty * omega).clamp(0.0, 1.0)
        return g * factor

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
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
                g_mod = self._attenuated(p, g)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                # shared routing: both moments see the attenuated gradient.
                routed = RoutedGradients(num=g_mod, den=g_mod)
                if self._routing == RoutingMode.OGP:
                    routed = RoutedGradients(num=g_mod, den=g)
                m.mul_(beta1).add_(routed.num, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(routed.den, routed.den, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]
                denom = (v / bc2).sqrt().add_(eps)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


class EWC(SharedRoutingPenalty):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    The importance map is the diagonal of the empirical Fisher on old-task
    data. Call ``estimate_fisher(data_iter, model, loss_fn)`` at the task
    boundary; the method refreshes ``Omega`` in place.
    """

    def estimate_fisher(self, data_iter, model, loss_fn, num_batches: int = 32) -> None:
        """Estimate the diagonal Fisher from a few batches of old-task data."""
        assert num_batches > 0, f"num_batches must be positive, got {num_batches}"
        fisher: Dict[int, torch.Tensor] = {
            id(p): torch.zeros_like(p) for p in model.parameters() if p.requires_grad
        }
        model.zero_grad(set_to_none=True)
        seen = 0
        for batch in data_iter:
            loss = loss_fn(model, batch)
            grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
            for p, g in zip(
                (p for p in model.parameters() if p.requires_grad),
                grads,
            ):
                fisher[id(p)] += g.detach().pow(2)
            seen += 1
            if seen >= num_batches:
                break
        assert seen > 0, "EWC.estimate_fisher got an empty data iterator"
        for k in fisher:
            fisher[k] = fisher[k] / max(seen, 1)
            m = fisher[k].max().clamp_min(1.0e-12)
            fisher[k] = fisher[k] / m  # normalise to [0,1] for lambda compatibility
        self.update_importance(fisher)


@dataclass
class SIState:
    """Running state for Synaptic Intelligence."""

    path_integral: Dict[int, torch.Tensor] = field(default_factory=dict)
    old_theta: Dict[int, torch.Tensor] = field(default_factory=dict)


class SI(SharedRoutingPenalty):
    """Synaptic Intelligence (Zenke et al., 2017).

    Path-integral importance: ``Omega_i = sum_t -g_i * delta_theta_i``
    accumulated along the training trajectory and normalised by the
    total parameter displacement. This class exposes ``track_step`` to
    be called after each ``.step()`` so that the path integral updates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._si = SIState()

    def begin_task(self, model) -> None:
        """Snapshot parameters and reset the running path integral."""
        self._si.path_integral = {
            id(p): torch.zeros_like(p) for p in model.parameters() if p.requires_grad
        }
        self._si.old_theta = {
            id(p): p.detach().clone() for p in model.parameters() if p.requires_grad
        }

    def track_step(self, model) -> None:
        """Update the path integral with the most recent step."""
        for p in model.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            # Per SI, contribution = -g * delta_theta. Since we use Adam,
            # we approximate delta_theta by the raw gradient times a unit
            # step size; the absolute scale is absorbed by the later
            # normalisation.
            contrib = -p.grad.detach() * (-p.grad.detach())
            self._si.path_integral[id(p)] += contrib

    def end_task(self, model, damping: float = 1.0e-3) -> None:
        """Materialise Omega from the path integral and hand it to the
        shared-routing step via ``update_importance``."""
        omega: Dict[int, torch.Tensor] = {}
        for p in model.parameters():
            if not p.requires_grad:
                continue
            pi = self._si.path_integral.get(id(p))
            if pi is None:
                continue
            delta = (p.detach() - self._si.old_theta[id(p)]).pow(2) + damping
            val = (pi / delta).clamp_min(0.0)
            m = val.max().clamp_min(1.0e-12)
            omega[id(p)] = val / m
        self.update_importance(omega)
