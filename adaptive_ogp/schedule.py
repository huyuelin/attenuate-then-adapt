"""Overlap-aware adaptive-strength controller.

Implements equation for ``alpha_t`` in Section 4 (Method) of the paper:

    alpha_t = alpha_max * (1 - bar_s_t)

with EMA-smoothed subspace-alignment signal

    bar_s_t = beta_s * bar_s_{t-1} + (1 - beta_s) * s_t,
    s_t     = ||U^T g_t||^2 / ||g_t||^2.

At bar_s_t ~ 0 (uncorrelated / low-overlap regimes) the controller saturates
to alpha_max and Adaptive-OGP reduces numerically to fixed-strength OGP.
At bar_s_t ~ 1 (high-overlap / stress regimes) alpha_t shrinks automatically
and avoids the fixed-strength-decoupled failure reported in Table 5.

The schedule's monotonicity property is verified in tests/test_schedule.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class OverlapAwareSchedule:
    """Overlap-aware adaptive-strength controller.

    Parameters
    ----------
    alpha_max: float
        Upper bound on alpha_t. Paper default is 0.5.
    beta_s: float
        EMA smoothing coefficient on the subspace-alignment signal.
    warmup_steps: int
        During warmup alpha_t is clamped to 0 so that the v_t EMA has a
        chance to fill with raw new-task statistics before any projection
        kicks in; the paper uses warmup = 0 by default.
    """

    alpha_max: float = 0.5
    beta_s: float = 0.98
    warmup_steps: int = 0
    _bar_s: float = 0.0
    _step: int = 0
    _history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert 0.0 <= self.alpha_max <= 1.0, f"alpha_max out of range: {self.alpha_max}"
        assert 0.0 <= self.beta_s < 1.0, f"beta_s out of range: {self.beta_s}"
        assert self.warmup_steps >= 0, f"warmup_steps negative: {self.warmup_steps}"

    def update(self, s_t: torch.Tensor | float) -> float:
        """Advance the controller by one step and return the current alpha_t.

        Parameters
        ----------
        s_t: scalar in [0,1]
            The instantaneous subspace-alignment signal at step t.
        """
        s_val = float(s_t.item() if torch.is_tensor(s_t) else s_t)
        assert 0.0 <= s_val <= 1.0 + 1.0e-6, f"s_t out of range: {s_val}"
        s_val = min(max(s_val, 0.0), 1.0)
        self._bar_s = self.beta_s * self._bar_s + (1.0 - self.beta_s) * s_val
        self._step += 1
        if self._step <= self.warmup_steps:
            alpha_t = 0.0
        else:
            alpha_t = self.alpha_max * (1.0 - self._bar_s)
        self._history.append(alpha_t)
        return alpha_t

    @property
    def bar_s(self) -> float:
        """Current smoothed overlap signal."""
        return self._bar_s

    @property
    def step(self) -> int:
        """Number of update() calls seen so far."""
        return self._step

    @property
    def history(self) -> List[float]:
        """Recorded alpha_t trajectory (one value per ``update`` call)."""
        return list(self._history)

    def reset(self) -> None:
        """Reset the EMA state; used at task boundaries if needed."""
        self._bar_s = 0.0
        self._step = 0
        self._history.clear()
