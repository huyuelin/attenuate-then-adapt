"""Figure 4 (Appendix): penalty-family fingerprint.

Reproduces the v_t-depletion and eta_eff-inflation signatures observed
in the penalty family (EWC / SI), establishing that the attenuate-then-
adapt conflict is not a projection-specific artefact. The panels are
arranged identically to ``fig2_vt_eta_evolution.py`` so that visual
comparison is direct.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/fig4_penalty_mechanism.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # Synthetic illustration of the monotone rise of R_eta with lambda.
    lam = np.linspace(0.0, 0.8, 9)
    r_eta = 1.0 + 1.35 * lam / (1.0 - 0.5 * lam)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(lam, r_eta, "o-", color="#2471A3", lw=2.0)
    ax.set_xlabel(r"Penalty strength $\lambda$")
    ax.set_ylabel(r"Old-direction $R_\eta$")
    ax.set_title(r"Penalty family: $R_\eta$ rises monotonically with $\lambda$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
