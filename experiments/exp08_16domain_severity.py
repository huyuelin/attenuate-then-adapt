"""Reproduces the long-sequence 16-domain severity table (Appendix).

Doubling the stream length from 8 to 16 tasks makes the realism gap
self-evident: the real overlap distribution alone produces enough
upper-tail transitions for the failure to appear without construction.
The paper reports a 4.5-4.8 unit gap between Adaptive-OGP and the
strongest shared-routing projection baseline at 16 domains.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._trainer import (  # noqa: E402
    factory_adam,
    factory_adaptive_ogp,
    forgetting_from_history,
    run_continual_demo,
)
from utils.logging import JSONLogger  # noqa: E402
from utils.seed import set_deterministic_seed  # noqa: E402


def run_demo(seed: int, out_dir: str) -> Dict[str, Any]:
    set_deterministic_seed(seed)
    methods = {
        "vanilla":       (factory_adam,          False),
        "adaptive_ogp":  (factory_adaptive_ogp,  True),
    }
    results: Dict[str, Any] = {}
    for name, (fac, needs) in methods.items():
        out = run_continual_demo(
            optimizer_factory=fac,
            num_tasks=8,        # demo uses 8 in place of 16 to stay fast
            tokens_per_task=2048,
            seq_len=16,
            batch_size=16,
            overlap=0.4,
            seed=seed,
            routing_needs_subspace=needs,
        )
        results[name] = {
            "final_ppl": out.final_perplexity,
            "forgetting_loss": forgetting_from_history(out.per_task_eval_history),
        }
    logger = JSONLogger(os.path.join(out_dir, "exp08_demo.jsonl"))
    with logger:
        logger.log({"experiment": "exp08_16domain_severity", "mode": "demo",
                    "note": "demo uses 8 tasks; full run uses 16",
                    "seed": seed, "results": results})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/exp08")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.demo:
        res = run_demo(args.seed, args.out_dir)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    raise NotImplementedError("See configs/16domain_256m.yaml and scripts/run_all_tables.sh.")


if __name__ == "__main__":
    raise SystemExit(main())
