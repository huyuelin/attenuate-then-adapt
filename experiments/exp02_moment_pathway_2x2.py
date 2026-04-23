"""Reproduces Table 2: moment-pathway 2x2 ablation.

The four conditions differ only in how g and tilde g are routed into
(m, v). All other hyperparameters are held fixed. In the paper the
observed ordering is

    vanilla (no protection)     >= v_only    shared   >=   ogp (best)

where ``vanilla`` has no attenuation, ``v_only`` inherits only the
denominator contribution (isolates the attenuate-then-adapt conflict),
``shared`` is the classical parameter-level routing, and ``ogp`` is
the repair advocated in the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_ogp.routing import RoutingMode  # noqa: E402
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
    conditions = {
        "vanilla": (RoutingMode.VANILLA, False),
        "v_only":  (RoutingMode.V_ONLY, True),
        "shared":  (RoutingMode.SHARED, True),
        "ogp":     (RoutingMode.OGP,    True),
    }
    results: Dict[str, Any] = {}
    for name, (mode, needs) in conditions.items():
        if mode == RoutingMode.VANILLA:
            factory = factory_adam
        else:
            factory = lambda m, s, _mode=mode: factory_adaptive_ogp(m, s, routing=_mode)
        out = run_continual_demo(
            optimizer_factory=factory,
            num_tasks=3,
            tokens_per_task=2048,
            seq_len=16,
            batch_size=16,
            overlap=0.35,
            seed=seed,
            routing_needs_subspace=needs,
        )
        results[name] = {
            "final_ppl": out.final_perplexity,
            "forgetting_loss": forgetting_from_history(out.per_task_eval_history),
        }
    logger = JSONLogger(os.path.join(out_dir, "exp02_demo.jsonl"))
    with logger:
        logger.log({"experiment": "exp02_moment_pathway_2x2", "mode": "demo",
                    "seed": seed, "results": results})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/exp02")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.demo:
        res = run_demo(args.seed, args.out_dir)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    raise NotImplementedError("See scripts/run_all_tables.sh for the full-scale run.")


if __name__ == "__main__":
    raise SystemExit(main())
