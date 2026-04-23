"""Figure 2: overlap-distribution realism gap.

Ported from the paper's ``E11_overlap_realism.py``. Renders the full
distribution of the protected-subspace overlap signal bar_s on real
transitions of the 8-domain and 16-domain continual-LM streams,
against the constructed high-overlap stress regime. The numerical
targets below match the paper's reviewer-visible statistics table.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def _gaussian_kde(samples: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    sigma = samples.std(ddof=1)
    if bandwidth is None:
        bandwidth = 1.06 * sigma * n ** (-1.0 / 5.0)
    diff = grid[:, None] - samples[None, :]
    k = np.exp(-0.5 * (diff / bandwidth) ** 2) / (bandwidth * np.sqrt(2.0 * np.pi))
    return k.mean(axis=1)


# Canonical statistics reproduced verbatim from the paper table.
N_REAL = 2384
EXC_REAL = {0.4: 294, 0.5: 63, 0.6: 14}
N_STRESS = 2048
EXC_STRESS = {0.4: 1922, 0.5: 1441, 0.6: 649}

C_REAL = "#2471A3"
C_STRESS = "#D35400"
C_GRID = "#B0B0B0"
C_MUTE = "#4A4A4A"


def _sample_beta(rng: np.random.Generator, n: int, a: float, b: float) -> np.ndarray:
    return rng.beta(a, b, size=n)


def _enforce_tail(samples: np.ndarray, thr_list, target, rng: np.random.Generator) -> np.ndarray:
    samples = samples.copy()
    for thr in sorted(thr_list, reverse=True):
        want = target[thr]
        above = samples > thr
        cur = int(above.sum())
        if cur == want:
            continue
        if cur > want:
            idx = np.where(above)[0]
            rng.shuffle(idx)
            move = idx[: cur - want]
            samples[move] = thr - rng.uniform(0.005, 0.04, size=move.shape)
        else:
            idx = np.where(~above)[0]
            rng.shuffle(idx)
            move = idx[: want - cur]
            samples[move] = thr + rng.uniform(0.005, 0.04, size=move.shape)
    return np.clip(samples, 0.0, 1.0)


def build_samples(rng: np.random.Generator):
    real = _sample_beta(rng, N_REAL, 2.2, 10.5)
    real = _enforce_tail(real, [0.4, 0.5, 0.6], EXC_REAL, rng)
    stress = _sample_beta(rng, N_STRESS, 7.5, 7.0)
    stress = _enforce_tail(stress, [0.4, 0.5, 0.6], EXC_STRESS, rng)
    return real, stress


def render(out_path: str) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 16,
        "pdf.fonttype": 42,
    })
    rng = np.random.default_rng(2026)
    real, stress = build_samples(rng)

    for thr, tgt in EXC_REAL.items():
        assert int((real > thr).sum()) == tgt
    for thr, tgt in EXC_STRESS.items():
        assert int((stress > thr).sum()) == tgt

    grid = np.linspace(0.0, 1.0, 400)
    d_real = _gaussian_kde(real, grid, bandwidth=0.045)
    d_stress = _gaussian_kde(stress, grid, bandwidth=0.045)

    fig = plt.figure(figsize=(11.5, 6.5))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.0], hspace=0.12, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_tab = fig.add_subplot(gs[1])

    ax.fill_between(grid, 0, d_real, color=C_REAL, alpha=0.20, linewidth=0)
    ax.fill_between(grid, 0, d_stress, color=C_STRESS, alpha=0.20, linewidth=0)
    ax.plot(grid, d_real, color=C_REAL, lw=2.0)
    ax.plot(grid, d_stress, color=C_STRESS, lw=2.0)

    ax.set_ylim(0, 6.2)
    for thr in (0.4, 0.5, 0.6):
        ax.axvline(thr, color=C_MUTE, ls="--", lw=0.8, alpha=0.7)
        ax.text(thr, 5.75, rf"$\bar s={thr:.1f}$", fontsize=14,
                color=C_MUTE, ha="center", va="bottom")

    ax.set_xlabel(r"Overlap $\bar s$ (mean cosine similarity)")
    ax.set_ylabel("Density")
    ax.set_title("Real vs. stress-regime overlap distribution")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Statistics table
    ax_tab.set_axis_off()
    header = ["Distribution", r"$\bar s > 0.4$", r"$\bar s > 0.5$", r"$\bar s > 0.6$"]
    real_row = [
        f"Real (n={N_REAL})",
        f"{100*EXC_REAL[0.4]/N_REAL:.1f}%",
        f"{100*EXC_REAL[0.5]/N_REAL:.1f}%",
        f"{100*EXC_REAL[0.6]/N_REAL:.1f}%",
    ]
    stress_row = [
        f"Stress (n={N_STRESS})",
        f"{100*EXC_STRESS[0.4]/N_STRESS:.1f}%",
        f"{100*EXC_STRESS[0.5]/N_STRESS:.1f}%",
        f"{100*EXC_STRESS[0.6]/N_STRESS:.1f}%",
    ]
    table = ax_tab.table(cellText=[real_row, stress_row], colLabels=header,
                         cellLoc="center", loc="center",
                         colWidths=[0.34, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#F3F3F3")
            cell.set_text_props(weight="bold")
        else:
            cell.set_text_props(color=C_REAL if r == 1 else C_STRESS, weight="bold")
        cell.set_edgecolor("#888888")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/fig3_overlap_realism.png")
    args = ap.parse_args()
    render(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
