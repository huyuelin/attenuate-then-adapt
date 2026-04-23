"""Baseline continual-learning methods for head-to-head comparison.

Covers the three families used in Table 7 (cross-family breadth):

* ``projection``: OGD, GPM, SGP, ROGO, FOPNG, Adam-NSCL.
* ``penalty``:    EWC (Fisher), SI (path-integral).
* ``replay``:     replay-gradient mixing with rho in {0.5%, 1%}.

All baselines follow the paper's shared-routing convention: the modified
gradient feeds both the numerator and denominator pathways. This is the
code-level definition of the attenuate-then-adapt conflict and is the
primary contrast to Adaptive-OGP.
"""

from baselines.projection import (
    AdamNSCL,
    FOPNG,
    GPM,
    OGD,
    ROGO,
    SGP,
    SharedRoutingProjection,
)
from baselines.penalty import EWC, SI, SharedRoutingPenalty
from baselines.replay import ReplayGradientMixing

__all__ = [
    "AdamNSCL",
    "FOPNG",
    "GPM",
    "OGD",
    "ROGO",
    "SGP",
    "SharedRoutingProjection",
    "EWC",
    "SI",
    "SharedRoutingPenalty",
    "ReplayGradientMixing",
]
