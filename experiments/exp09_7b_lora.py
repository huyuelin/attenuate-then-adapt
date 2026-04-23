"""Reproduces the 7B LoRA TRACE experiment.

This driver is intentionally a stub at the top level: the full run
requires 8xA100 and a TRACE install (see ``scripts/run_7b_lora.sh``).
The stub still exposes a schema-compatible entry point so that a
reviewer can see how the Adaptive-OGP optimizer is wired into LoRA.

A toy equivalent (``--demo``) runs the tiny model stream at the same
routing and schedule settings as the full 7B config; it is not intended
as a quantitative stand-in for the 7B result but only as a smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.trace_benchmark import TRACEBenchmarkStub  # noqa: E402
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
    stub = TRACEBenchmarkStub()
    # stub.require_local() is intentionally *not* called in the demo.
    results: Dict[str, Any] = {"trace_tasks": [t.name for t in stub.task_list()]}
    for name, (fac, needs) in {
        "vanilla":      (factory_adam,          False),
        "adaptive_ogp": (factory_adaptive_ogp,  True),
    }.items():
        out = run_continual_demo(
            optimizer_factory=fac,
            num_tasks=4,
            tokens_per_task=2048,
            seq_len=16,
            batch_size=16,
            overlap=0.3,
            seed=seed,
            routing_needs_subspace=needs,
        )
        results[name] = {
            "final_ppl": out.final_perplexity,
            "forgetting_loss": forgetting_from_history(out.per_task_eval_history),
        }
    logger = JSONLogger(os.path.join(out_dir, "exp09_demo.jsonl"))
    with logger:
        logger.log({"experiment": "exp09_7b_lora", "mode": "demo",
                    "note": "toy stand-in; see scripts/run_7b_lora.sh for real run",
                    "seed": seed, "results": results})
    return results


def run_full() -> None:
    """Hook for the full 7B LoRA run; intentionally not implemented."""
    raise NotImplementedError(
        "The 7B LoRA run requires a GPU cluster and a TRACE install. "
        "See scripts/run_7b_lora.sh for the entry point, and "
        "configs/7b_lora.yaml for the configuration."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/exp09")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.demo:
        res = run_demo(args.seed, args.out_dir)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    run_full()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
