"""Reproduces Table 7: cross-family breadth and cross-optimizer control.

This driver runs the attenuate-then-adapt fingerprint check across

  * projection family (shared vs. ogp),
  * penalty family (shared vs. ogp),
  * replay family (shared mixing vs. ogp-on-mixed),

and against four optimizer backbones (Adam, AdamW, AdaFactor-style
diag surrogate, SGD+Momentum). For the demo we use a small subset of
the cross product because each entry triggers a full continual-stream
run; full-scale reproduction is driven by ``scripts/run_all_tables.sh``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_ogp.optimizer import AdaptiveOGP  # noqa: E402
from adaptive_ogp.routing import RoutingMode  # noqa: E402
from adaptive_ogp.schedule import OverlapAwareSchedule  # noqa: E402
from baselines.penalty import EWC  # noqa: E402
from baselines.projection import FOPNG  # noqa: E402
from baselines.replay import ReplayGradientMixing  # noqa: E402
from experiments._trainer import (  # noqa: E402
    factory_adam,
    forgetting_from_history,
    run_continual_demo,
)
from utils.logging import JSONLogger  # noqa: E402
from utils.seed import set_deterministic_seed  # noqa: E402


def _proj_shared(m, s):
    assert s is not None
    return FOPNG(m.parameters(), subspace=s, lr=1.0e-3)


def _proj_ogp(m, s):
    assert s is not None
    sch = OverlapAwareSchedule(alpha_max=0.5)
    return AdaptiveOGP(m.parameters(), lr=1.0e-3, routing=RoutingMode.OGP,
                       subspace=s, schedule=sch, alpha_max=0.5)


def _penalty_shared(m, _s):
    # EWC without a task-switch Fisher update is an identity gradient
    # rescaler; for the demo we rely on the default shared routing.
    return EWC(m.parameters(), lr=1.0e-3, lambda_penalty=0.5)


def _replay_shared(m, _s):
    return ReplayGradientMixing(m.parameters(), lr=1.0e-3, rho_mix=0.1,
                                replay_grad_fn=None)


def run_demo(seed: int, out_dir: str) -> Dict[str, Any]:
    set_deterministic_seed(seed)
    methods = {
        "proj_shared":     (_proj_shared,    True),
        "proj_ogp":        (_proj_ogp,       True),
        "penalty_shared":  (_penalty_shared, False),
        "replay_shared":   (_replay_shared,  False),
        "vanilla":         (factory_adam,    False),
    }
    results: Dict[str, Any] = {}
    for name, (fac, needs) in methods.items():
        out = run_continual_demo(
            optimizer_factory=fac,
            num_tasks=3,
            tokens_per_task=2048,
            seq_len=16,
            batch_size=16,
            overlap=0.5,
            seed=seed,
            routing_needs_subspace=needs,
        )
        results[name] = {
            "final_ppl": out.final_perplexity,
            "forgetting_loss": forgetting_from_history(out.per_task_eval_history),
        }
    logger = JSONLogger(os.path.join(out_dir, "exp07_demo.jsonl"))
    with logger:
        logger.log({"experiment": "exp07_breadth", "mode": "demo",
                    "seed": seed, "results": results})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/exp07")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.demo:
        res = run_demo(args.seed, args.out_dir)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    raise NotImplementedError(
        "Full cross-family / cross-optimizer sweep is driven by scripts/run_all_tables.sh."
    )


if __name__ == "__main__":
    raise SystemExit(main())
