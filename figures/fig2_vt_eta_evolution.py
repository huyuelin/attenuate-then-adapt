"""Figure 1: v_t depletion and R_eta inflation time-series.

Generates a two-panel figure matching the paper's fingerprint plot of
the attenuate-then-adapt conflict. Panel A shows the old-direction
second-moment energy E_old(t) diverging between shared-routing and
OGP after the task switch; Panel B shows the effective learning-rate
ratio R_eta(t) inflating correspondingly under shared routing.

Because full 256M HOPE training exceeds CPU demo compute, this figure
runs on the toy model and two-task stream; the qualitative fingerprint
is identical.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_ogp.optimizer import AdaptiveOGP  # noqa: E402
from adaptive_ogp.probes import EtaEffProbe, VtEnergyProbe  # noqa: E402
from adaptive_ogp.routing import RoutingMode  # noqa: E402
from adaptive_ogp.schedule import OverlapAwareSchedule  # noqa: E402
from adaptive_ogp.subspace import SubspaceManager  # noqa: E402
from benchmarks.continual_lm_8domain import batch_iter, build_toy_stream  # noqa: E402
from experiments._trainer import ToyLanguageModel, cross_entropy_loss  # noqa: E402


def _collect_fingerprint(routing: RoutingMode, seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    model = ToyLanguageModel()
    mgr = SubspaceManager(rank=8, buffer_capacity=64)
    opt = AdaptiveOGP(model.parameters(), lr=1.0e-3, routing=routing,
                      subspace=mgr, schedule=OverlapAwareSchedule(alpha_max=0.5),
                      alpha_max=0.5)
    vt_probe = VtEnergyProbe(subspace=mgr)
    eta_probe = EtaEffProbe(subspace=mgr)
    stream = build_toy_stream(num_tasks=2, tokens_per_task=2048, seq_len=16,
                              overlap=0.5, seed=seed)

    # Task A: train, collect gradients for the basis.
    train_a, _ = stream[0]
    for step, batch in enumerate(batch_iter(train_a, 16, 16)):
        opt.zero_grad(set_to_none=True)
        cross_entropy_loss(model, batch).backward()
        for p in model.parameters():
            if p.grad is not None:
                mgr.collect(p, p.grad.detach())
        opt.step()
    opt.on_task_switch()

    # Task B: probe after every step.
    train_b, _ = stream[1]
    for step, batch in enumerate(batch_iter(train_b, 16, 16)):
        opt.zero_grad(set_to_none=True)
        cross_entropy_loss(model, batch).backward()
        opt.step()
        vt_probe.measure(opt)
        eta_probe.measure(opt)
    return vt_probe.history, eta_probe.history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/fig2_vt_eta_evolution.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shared_vt, shared_eta = _collect_fingerprint(RoutingMode.SHARED, seed=args.seed)
    ogp_vt, ogp_eta = _collect_fingerprint(RoutingMode.OGP, seed=args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axes[0].plot(np.array(shared_vt), label="shared-routing", color="#D35400", lw=2.0)
    axes[0].plot(np.array(ogp_vt),    label="OGP (decoupled)",  color="#2471A3", lw=2.0)
    axes[0].set_title("Old-direction second-moment energy")
    axes[0].set_xlabel("step on task B")
    axes[0].set_ylabel(r"$E_{\mathrm{old}}(t)$ (proxy)")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.array(shared_eta), label="shared-routing", color="#D35400", lw=2.0)
    axes[1].plot(np.array(ogp_eta),    label="OGP (decoupled)",  color="#2471A3", lw=2.0)
    axes[1].set_title(r"Old-direction $R_\eta$ ratio")
    axes[1].set_xlabel("step on task B")
    axes[1].set_ylabel(r"$R_\eta(t)$ (proxy)")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
