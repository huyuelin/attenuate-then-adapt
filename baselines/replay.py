"""Replay-gradient mixing baseline.

Paper reference: Section ``Cross-family breadth'', Table 7 row ``replay''.
At each step we draw a batch from a bounded-size replay buffer (rho ~ 0.5%
of the training data), compute an old-task gradient ``g_r``, and mix:

    g_mix_t = (1 - rho_mix) * g_t + rho_mix * g_r.

Shared routing feeds ``g_mix`` into both m and v. The paper shows that
under this shared routing the method still trails Adaptive-OGP by
~2.2 units on 8-domain and ~2.6 units on 16-domain.

The implementation provided here is a self-contained wrapper around
Adam that expects the user to supply an iterator over replay batches
plus a ``replay_grad_fn`` to compute ``g_r``. This keeps the code
model-agnostic; the experiment scripts drive it.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class ReplayGradientMixing(Optimizer):
    """Adam with mixed-gradient shared routing.

    Parameters
    ----------
    params, lr, betas, eps, weight_decay: Adam hyperparameters.
    rho_mix: float in [0,1]
        Weight of the replay gradient in the mix. The paper varies rho
        in {0.5%, 1%} buffer sizes; ``rho_mix`` is the *mixing* coefficient
        used inside the step, which is a separate knob.
    replay_grad_fn: optional callable
        Takes no arguments and returns a list of replay gradients in the
        same order as ``self.param_groups[0]['params']``. If None, the
        method falls back to a zero replay gradient (no effect).
    """

    def __init__(
        self,
        params,
        lr: float = 1.0e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        rho_mix: float = 0.5,
        replay_grad_fn: Optional[Callable[[], list]] = None,
    ) -> None:
        assert 0.0 <= rho_mix <= 1.0, f"rho_mix out of range: {rho_mix}"
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rho_mix = rho_mix
        self.replay_grad_fn = replay_grad_fn

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        replay_grads = None
        if self.replay_grad_fn is not None:
            replay_grads = self.replay_grad_fn()
            assert replay_grads is not None, "replay_grad_fn returned None"

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    idx += 1
                    continue
                g = p.grad
                assert torch.isfinite(g).all()
                if replay_grads is not None:
                    g_r = replay_grads[idx]
                    assert g_r.shape == g.shape, (
                        f"replay gradient shape {tuple(g_r.shape)} != {tuple(g.shape)}"
                    )
                    g_mix = (1.0 - self.rho_mix) * g + self.rho_mix * g_r
                else:
                    g_mix = g
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                # shared routing: g_mix enters both moments.
                m.mul_(beta1).add_(g_mix, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_mix, g_mix, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]
                denom = (v / bc2).sqrt().add_(eps)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(m, denom, value=-lr / bc1)
                idx += 1
        return loss
