"""Reproduces Table 4: eta_eff-matching intervention.

We manually equalise the old-direction effective learning-rate ratio
R_eta between parameter-level (shared) and OGP routings via the scalar
c_+ multiplier on the squared gradient fed to v_t (see
``adaptive_ogp.interventions.EtaMatchingIntervention``). The paper
reports that matching R_eta closes the 1.9-unit forgetting gap to
~0.3-0.4 units, with Pearson r = 0.98 across four rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_ogp.interventions import EtaMatchingIntervention  # noqa: E402
from adaptive_ogp.routing import RoutingMode, apply_routing, project_out  # noqa: E402
from adaptive_ogp.subspace import SubspaceManager  # noqa: E402
from benchmarks.continual_lm_8domain import batch_iter, build_toy_stream  # noqa: E402
from experiments._trainer import (  # noqa: E402
    ToyLanguageModel,
    cross_entropy_loss,
    forgetting_from_history,
)
from utils.logging import JSONLogger  # noqa: E402
from utils.seed import set_deterministic_seed  # noqa: E402


class EtaMatchedOptimizer(torch.optim.Optimizer):
    """Adam with scalar rescaling on the squared-gradient input to v_t."""

    def __init__(
        self,
        params,
        subspace: SubspaceManager,
        routing: RoutingMode,
        intervention: EtaMatchingIntervention,
        lr: float = 1.0e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1.0e-8,
        alpha_max: float = 0.5,
    ) -> None:
        assert lr > 0
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.subspace = subspace
        self.routing = routing
        self.intervention = intervention
        self.alpha_max = alpha_max

    def on_task_switch(self) -> None:
        self.subspace.build_bases()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        alpha = self.alpha_max if self.subspace.has_any_basis() else 0.0
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert torch.isfinite(g).all()
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                basis, sigma_hat = self.subspace.basis_for(p)
                if basis is None:
                    g_mod = g
                else:
                    U = basis.to(device=g.device, dtype=g.dtype)
                    sig = sigma_hat.to(device=g.device, dtype=g.dtype)
                    g_mod = project_out(g, U, sig, alpha)
                routed = apply_routing(g, g_mod, self.routing)
                routed = self.intervention.apply(routed)
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(routed.num, alpha=1 - b1)
                v.mul_(b2).addcmul_(routed.den, routed.den, value=1 - b2)
                bc1 = 1 - b1 ** st["step"]
                bc2 = 1 - b2 ** st["step"]
                denom = (v / bc2).sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)


def _run(routing: RoutingMode, c_plus: float, seed: int) -> float:
    torch.manual_seed(seed)
    model = ToyLanguageModel()
    mgr = SubspaceManager(rank=8, buffer_capacity=64)
    opt = EtaMatchedOptimizer(
        model.parameters(), mgr, routing,
        EtaMatchingIntervention(c_plus=c_plus),
    )
    stream = build_toy_stream(num_tasks=3, tokens_per_task=2048, seq_len=16,
                              overlap=0.35, seed=seed)
    history: List[List[float]] = [[] for _ in stream]
    for ti, (train, _) in enumerate(stream):
        for step, batch in enumerate(batch_iter(train, 16, 16)):
            opt.zero_grad(set_to_none=True)
            cross_entropy_loss(model, batch).backward()
            if step >= (len(train) // (16 * 16)) - 8:
                for p in model.parameters():
                    if p.grad is not None:
                        mgr.collect(p, p.grad.detach())
            opt.step()
        if ti < len(stream) - 1:
            opt.on_task_switch()
        for ei, (_, ev) in enumerate(stream[: ti + 1]):
            model.eval()
            with torch.no_grad():
                ls = [float(cross_entropy_loss(model, b).item())
                      for b in batch_iter(ev, 16, 16, shuffle=False)]
            model.train()
            history[ei].append(sum(ls) / max(len(ls), 1))
    return forgetting_from_history(history)


def run_demo(seed: int, out_dir: str) -> Dict[str, Any]:
    set_deterministic_seed(seed)
    conditions = {
        "shared_c1":      (RoutingMode.SHARED, 1.0),
        "shared_cplus":   (RoutingMode.SHARED, 4.3),   # paper's c_+ proxy
        "ogp_c1":         (RoutingMode.OGP,    1.0),
        "ogp_cminus":     (RoutingMode.OGP,    0.9),   # symmetric downscale
    }
    results = {k: {"forgetting_loss": _run(r, c, seed)} for k, (r, c) in conditions.items()}
    logger = JSONLogger(os.path.join(out_dir, "exp04_demo.jsonl"))
    with logger:
        logger.log({"experiment": "exp04_eta_matching", "mode": "demo",
                    "seed": seed, "results": results})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/exp04")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.demo:
        res = run_demo(args.seed, args.out_dir)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    raise NotImplementedError("See scripts/run_all_tables.sh for the full-scale run.")


if __name__ == "__main__":
    raise SystemExit(main())
