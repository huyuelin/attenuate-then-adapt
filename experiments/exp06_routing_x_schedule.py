"""Reproduces Table 6: routing x schedule 2x2.

Crossing ``routing in {shared, ogp}`` with ``schedule in {fixed, adaptive}``
quantifies the routing and the schedule as independent contributors:

    shared + fixed      (near-vanilla)
    shared + adaptive   (partial recovery)
    ogp + fixed         (below-vanilla under stress)
    ogp + adaptive      (repair)
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
from experiments._trainer import (  # noqa: E402
    forgetting_from_history,
    run_continual_demo,
)
from utils.logging import JSONLogger  # noqa: E402
from utils.seed import set_deterministic_seed  # noqa: E402


def _factory(routing: RoutingMode, adaptive: bool):
    def _f(m, s):
        assert s is not None
        sch = OverlapAwareSchedule(alpha_max=0.5) if adaptive else None
        return AdaptiveOGP(m.parameters(), lr=1.0e-3, routing=routing,
                           subspace=s, schedule=sch, alpha_max=0.5)
    return _f


def run_demo(seed: int, out_dir: str) -> Dict[str, Any]:
    set_deterministic_seed(seed)
    conditions = {
        "shared_fixed":   (RoutingMode.SHARED, False),
        "shared_adapt":   (RoutingMode.SHARED, True),
        "ogp_fixed":      (RoutingMode.OGP,    False),
        "ogp_adapt":      (RoutingMode.OGP,    True),
    }
    results: Dict[str, Any] = {}
    for name, (mode, adapt) in conditions.items():
        out = run_continual_demo(
            optimizer_factory=_factory(mode, adapt),
            num_tasks=3,
            tokens_per_task=2048,
            seq_len=16,
            batch_size=16,
            overlap=0.6,
            seed=seed,
        )
        results[name] = {
            "final_ppl": out.final_perplexity,
            "forgetting_loss": forgetting_from_history(out.per_task_eval_history),
        }
    logger = JSONLogger(os.path.join(out_dir, "exp06_demo.jsonl"))
    with logger:
        logger.log({"experiment": "exp06_routing_x_schedule", "mode": "demo",
                    "seed": seed, "results": results})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/exp06")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.demo:
        res = run_demo(args.seed, args.out_dir)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    raise NotImplementedError("See scripts/run_all_tables.sh for the full-scale run.")


if __name__ == "__main__":
    raise SystemExit(main())
