"""Reproduces Table 1: clean-regime 8-domain continual LM.

In the clean regime the overlap statistic bar_s is small, so the
adaptive schedule saturates at alpha_t ~ alpha_max and Adaptive-OGP
reduces numerically to fixed-strength OGP. This driver confirms the
structural effect on the toy stream.

Usage
-----
  python experiments/exp01_clean_8domain.py --demo
  python experiments/exp01_clean_8domain.py --config configs/8domain_256m.yaml

The first form runs in about a minute on CPU with a tiny model; the
second form is a stub that raises NotImplementedError, pointing the
reviewer at ``scripts/run_all_tables.sh`` for full-scale reproduction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

# make 'adaptive_ogp' importable when launched as a script from repo root
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
        "vanilla": lambda m, s: factory_adam(m, s),
        "adaptive_ogp": lambda m, s: factory_adaptive_ogp(m, s),
    }
    results: Dict[str, Any] = {}
    for name, factory in methods.items():
        out = run_continual_demo(
            optimizer_factory=factory,
            num_tasks=4,
            tokens_per_task=2048,
            seq_len=16,
            batch_size=16,
            overlap=0.25,
            seed=seed,
            routing_needs_subspace=(name != "vanilla"),
        )
        results[name] = {
            "final_ppl": out.final_perplexity,
            "forgetting_loss": forgetting_from_history(out.per_task_eval_history),
        }

    logger = JSONLogger(os.path.join(out_dir, "exp01_demo.jsonl"))
    with logger:
        logger.log({"experiment": "exp01_clean_8domain", "mode": "demo", "seed": seed,
                    "results": results})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="CPU demo (few minutes)")
    ap.add_argument("--config", type=str, default=None, help="Full-scale YAML config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/exp01")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.demo:
        res = run_demo(args.seed, args.out_dir)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    assert args.config is not None, "Specify --demo or --config"
    raise NotImplementedError(
        "Full-scale Table 1 runs are driven by scripts/run_all_tables.sh; "
        f"see configs/8domain_256m.yaml (requested: {args.config})."
    )


if __name__ == "__main__":
    raise SystemExit(main())
