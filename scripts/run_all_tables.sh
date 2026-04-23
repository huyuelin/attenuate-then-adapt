#!/usr/bin/env bash
# Drive the full-scale reproduction of every main table.
#
# Prerequisites:
#   * 8xA100 80GB or equivalent (7B LoRA run).
#   * Datasets downloaded via scripts/download_benchmarks.sh.
#   * TRACE benchmark extracted and TRACE_ROOT exported.
#
# Each experiment script exposes a ``--config`` entry that is currently
# a stub (NotImplementedError). The hooks below are provided as the
# canonical command list; fill them in against your in-house trainer
# when porting to a real cluster.

set -e

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"

: "${DATA_ROOT:=./data/8domain}"
: "${TRACE_ROOT:=./data/trace}"

echo "[full] Table 1: clean 8-domain"
python experiments/exp01_clean_8domain.py --config configs/8domain_256m.yaml \
    --out-dir runs/full/exp01

echo "[full] Table 2: moment-pathway 2x2"
python experiments/exp02_moment_pathway_2x2.py --config configs/8domain_256m.yaml \
    --out-dir runs/full/exp02

echo "[full] Table 3: denominator-only"
python experiments/exp03_denominator_intervention.py --config configs/8domain_256m.yaml \
    --out-dir runs/full/exp03

echo "[full] Table 4: eta-matching"
python experiments/exp04_eta_matching.py --config configs/8domain_256m.yaml \
    --out-dir runs/full/exp04

echo "[full] Table 5: stress-overlap"
python experiments/exp05_stress_overlap.py --config configs/8domain_256m.yaml \
    --out-dir runs/full/exp05

echo "[full] Table 6: routing x schedule"
python experiments/exp06_routing_x_schedule.py --config configs/8domain_256m.yaml \
    --out-dir runs/full/exp06

echo "[full] Table 7: cross-family breadth"
python experiments/exp07_breadth.py --config configs/8domain_256m.yaml \
    --out-dir runs/full/exp07

echo "[full] Appendix: 16-domain severity"
python experiments/exp08_16domain_severity.py --config configs/16domain_256m.yaml \
    --out-dir runs/full/exp08

echo "[full] Appendix: 7B LoRA TRACE"
python experiments/exp09_7b_lora.py --config configs/7b_lora.yaml \
    --out-dir runs/full/exp09
