"""Reproduces Table 5: stress-overlap regime.

We combine two adversarial conditions: high inter-task overlap (set via
the toy-stream ``overlap`` argument) and a non-adaptive schedule
(``OverlapAwareSchedule`` disabled, fixed alpha_max in effect). Under
this regime every shared-routing baseline collapses near vanilla, a
naive fixed-strength decoupled variant falls below vanilla, and only
Adaptive-OGP (with the overlap-aware schedule enabled) remains stable.
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
from baselines.projection import FOPNG, AdamNSCL  # noqa: E402
from experiments._trainer import (  # noqa: E402
    factory_adam,
    forgetting_from_history,
    run_continual_demo,
)
from utils.logging import JSONLogger  # noqa: E402
from utils.seed import set_deterministic_seed  # noqa: E402


def _factory_fopng(m, s):
    assert s is not None
    return FOPNG(m.parameters(), subspace=s, lr=1.0e-3)


def _factory_adamnscl(m, s):
    assert s is not None
    return AdamNSCL(m.parameters(), subspace=s, lr=1.0e-3)


def _factory_fixed_ogp(m, s):
    assert s is not None
    # fixed-strength decoupled routing: Adaptive-OGP with schedule disabled.
    return AdaptiveOGP(m.parameters(), lr=1.0e-3, routing=RoutingMode.OGP,
                       subspace=s, schedule=None, alpha_max=0.5)


def _factory_adaptive_ogp(m, s):
    assert s is not None
    sch = OverlapAwareSchedule(alpha_max=0.5)
    return AdaptiveOGP(m.parameters(), lr=1.0e-3, routing=RoutingMode.OGP,
                       subspace=s, schedule=sch, alpha_max=0.5)


def run_demo(seed: int, out_dir: str) -> Dict[str, Any]:
    set_deterministic_seed(seed)
    methods = {
        "vanilla":       (factory_adam,           False),
        "fopng":         (_factory_fopng,         True),
        "adam_nscl":     (_factory_adamnscl,      True),
        "fixed_ogp":     (_factory_fixed_ogp,     True),
        "adaptive_ogp":  (_factory_adaptive_ogp,  True),
    }
    results: Dict[str, Any] = {}
    for name, (fac, needs) in methods.items():
        out = run_continual_demo(
            optimizer_factory=fac,
            num_tasks=3,
            tokens_per_task=2048,
            seq_len=16,
            batch_size=16,
            overlap=0.65,     # stress regime (realistic upper tail)
            seed=seed,
            routing_needs_subspace=needs,
        )
        results[name] = {
            "final_ppl": out.final_perplexity,
            "forgetting_loss": forgetting_from_history(out.per_task_eval_history),
        }
    logger = JSONLogger(os.path.join(out_dir, "exp05_demo.jsonl"))
    with logger:
        logger.log({"experiment": "exp05_stress_overlap", "mode": "demo",
                    "seed": seed, "results": results})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/exp05")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.demo:
        res = run_demo(args.seed, args.out_dir)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    raise NotImplementedError("See scripts/run_all_tables.sh for the full-scale run.")


if __name__ == "__main__":
    raise SystemExit(main())
