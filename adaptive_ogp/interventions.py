"""Controlled interventions used for causal isolation.

These wrap the optimizer step with surgical rescalings on the v_t
pathway, matching the constructions used in Tables 3 and 4:

* ``DenominatorOnlyIntervention`` (Table 3): holds m_t at the projected
  gradient and overrides only the denominator. Four v-pathway variants
  are provided: ``projected``, ``rescaled``, ``interpolated``, ``raw``.
* ``EtaMatchingIntervention`` (Table 4): rescales the squared gradient
  that enters v_t by a scalar c_+ (or c_-) so that the old-direction
  R_eta ratio is numerically equalised across routings; the paper
  reports c_+ ~ 4.3 for parameter-level and 0.9 for OGP.

These classes are intentionally minimalistic: they are wrappers that
replace ``RoutedGradients.den`` just before the v_t EMA update. The
experiment scripts drive them by subclassing ``AdaptiveOGP.step``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from adaptive_ogp.routing import RoutedGradients


@dataclass
class DenominatorOnlyIntervention:
    """Replace the denominator pathway while keeping m_t untouched.

    Parameters
    ----------
    mode: one of ``"projected"``, ``"rescaled"``, ``"interpolated"``, ``"raw"``.
    c_minus: scalar used in the ``rescaled`` variant (Table 3 surrogate).
        Default 0.25 corresponds to (1 - alpha)^2 at alpha = 0.5.
    mix: scalar used in the ``interpolated`` variant in [0,1];
        Table 3 uses ``0.75 v_raw + 0.25 v_proj``.
    """

    mode: str = "rescaled"
    c_minus: float = 0.25
    mix: float = 0.25

    def __post_init__(self) -> None:
        assert self.mode in {"projected", "rescaled", "interpolated", "raw"}, (
            f"unknown denominator mode {self.mode!r}"
        )
        assert 0.0 <= self.mix <= 1.0, f"mix out of range: {self.mix}"
        assert self.c_minus > 0.0, f"c_minus must be positive, got {self.c_minus}"

    def apply(
        self,
        routed: RoutedGradients,
        raw_grad: torch.Tensor,
        modified_grad: torch.Tensor,
    ) -> RoutedGradients:
        if self.mode == "projected":
            new_den = modified_grad
        elif self.mode == "rescaled":
            # manually rescale so that v_t steady state matches the
            # shared-routing surrogate; equivalent to scaling g^2 by c_-.
            scale = float(self.c_minus) ** 0.5
            new_den = raw_grad * scale
        elif self.mode == "interpolated":
            new_den = (1.0 - self.mix) * raw_grad + self.mix * modified_grad
        elif self.mode == "raw":
            new_den = raw_grad
        else:
            raise AssertionError(f"unreachable mode {self.mode!r}")
        return RoutedGradients(num=routed.num, den=new_den)


@dataclass
class EtaMatchingIntervention:
    """Rescale squared-gradient input to v_t so that R_eta is equalised.

    Parameters
    ----------
    c_plus: scalar (>0) multiplying the squared gradient fed into v_t.
        The paper uses c_+ ~ 4.3 for parameter-level routing (to push
        R_eta up to 1.08) and 0.9 for OGP (to raise R_eta to 1.11).
    """

    c_plus: float = 1.0

    def __post_init__(self) -> None:
        assert self.c_plus > 0.0, f"c_plus must be positive, got {self.c_plus}"

    def apply(self, routed: RoutedGradients) -> RoutedGradients:
        scale = float(self.c_plus) ** 0.5
        return RoutedGradients(num=routed.num, den=routed.den * scale)


def make_apply_hook(
    intervention: Callable[..., RoutedGradients],
) -> Callable[[RoutedGradients, torch.Tensor, torch.Tensor], RoutedGradients]:
    """Adapt a concrete intervention into the hook signature that the
    experiment driver expects (``(routed, raw, modified) -> routed``)."""

    def _hook(routed: RoutedGradients, raw: torch.Tensor, modified: torch.Tensor):
        if isinstance(intervention, DenominatorOnlyIntervention):
            return intervention.apply(routed, raw, modified)
        if isinstance(intervention, EtaMatchingIntervention):
            return intervention.apply(routed)
        raise AssertionError(f"unknown intervention type {type(intervention)}")

    return _hook
