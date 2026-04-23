"""Overview figure placeholder (Figure of three-panel story).

The actual paper asset is a hand-drawn schematic (provided as a PDF in
the submission). This script renders a simple caption-matched sketch
with matplotlib so that the repository is self-contained; reviewers who
want the paper-quality version should consult the submitted PDF.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    titles = [
        "Failure:\nshared-routing collapses to vanilla",
        "Diagnosis:\nattenuate-then-adapt conflict",
        "Repair:\nAdaptive-OGP (decoupled + schedule)",
    ]
    for ax, title in zip(axes, titles):
        ax.set_axis_off()
        ax.text(0.5, 0.55, title, ha="center", va="center",
                fontsize=14, family="serif")
        ax.text(0.5, 0.20, "see paper Figure 1", ha="center", va="center",
                fontsize=10, color="#555555", style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/fig1_overview.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    render(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
