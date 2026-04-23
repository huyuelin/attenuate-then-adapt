"""Adaptive-OGP: reference implementation companion to the NeurIPS 2026
submission "Hidden Failure Modes of Gradient Modification under Adam in
Continual Learning, and Adaptive Decoupled Moment Routing as a Repair".

Public API
----------
``AdaptiveOGP``
    Core optimizer (``torch.optim.Optimizer`` subclass) implementing the
    decoupled moment routing described in Algorithm 1 of the paper.
``SubspaceManager``
    Groupwise SVD basis extraction and protected-direction bookkeeping.
``OverlapAwareSchedule``
    Adaptive strength controller ``alpha_t = alpha_max * (1 - bar_s_t)``.
``VtEnergyProbe``, ``EtaEffProbe``
    Diagnostic probes reported in Figure 1 and Tables 3 to 4.

All method-level symbols in this package use only the vocabulary of the
paper: ``attenuate-then-adapt``, ``shared-routing``, ``decoupled-routing``,
``overlap-aware schedule``, ``OGP``, and ``Adaptive-OGP``.
"""

from adaptive_ogp.optimizer import AdaptiveOGP
from adaptive_ogp.routing import RoutingMode, apply_routing
from adaptive_ogp.schedule import OverlapAwareSchedule
from adaptive_ogp.subspace import SubspaceManager, extract_subspace_basis
from adaptive_ogp.probes import EtaEffProbe, SubspaceAlignmentProbe, VtEnergyProbe
from adaptive_ogp.interventions import DenominatorOnlyIntervention, EtaMatchingIntervention

__all__ = [
    "AdaptiveOGP",
    "RoutingMode",
    "apply_routing",
    "OverlapAwareSchedule",
    "SubspaceManager",
    "extract_subspace_basis",
    "EtaEffProbe",
    "SubspaceAlignmentProbe",
    "VtEnergyProbe",
    "DenominatorOnlyIntervention",
    "EtaMatchingIntervention",
]

__version__ = "0.1.0"
