#!/usr/bin/env bash
# Minute-scale smoke test. Runs every experiment's --demo entry point
# and emits structured JSON into runs/. Designed to exercise the full
# code-path tree on CPU; not a quantitative reproduction.
set -e

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"

mkdir -p runs

echo "[demo] exp01 clean 8-domain"
python experiments/exp01_clean_8domain.py --demo --out-dir runs/exp01

echo "[demo] exp02 moment-pathway 2x2"
python experiments/exp02_moment_pathway_2x2.py --demo --out-dir runs/exp02

echo "[demo] exp03 denominator-only intervention"
python experiments/exp03_denominator_intervention.py --demo --out-dir runs/exp03

echo "[demo] exp04 eta-matching intervention"
python experiments/exp04_eta_matching.py --demo --out-dir runs/exp04

echo "[demo] exp05 stress-overlap regime"
python experiments/exp05_stress_overlap.py --demo --out-dir runs/exp05

echo "[demo] exp06 routing x schedule"
python experiments/exp06_routing_x_schedule.py --demo --out-dir runs/exp06

echo "[demo] exp07 cross-family breadth"
python experiments/exp07_breadth.py --demo --out-dir runs/exp07

echo "[demo] exp08 16-domain severity"
python experiments/exp08_16domain_severity.py --demo --out-dir runs/exp08

echo "[demo] exp09 7B LoRA (toy stand-in)"
python experiments/exp09_7b_lora.py --demo --out-dir runs/exp09

echo "[demo] figures"
python figures/fig1_overview.py --out runs/fig1_overview.png
python figures/fig3_overlap_realism.py --out runs/fig3_overlap_realism.png

echo "[demo] done. Outputs under runs/."
