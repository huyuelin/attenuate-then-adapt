# Adaptive-OGP: Hidden Failure Modes of Gradient Modification under Adam in Continual Learning

Reference implementation companion to the NeurIPS 2026 submission
"Hidden Failure Modes of Gradient Modification under Adam in Continual
Learning, and Adaptive Decoupled Moment Routing as a Repair."

Anonymous authors, NeurIPS 2026 double-blind submission.

This repository will be de-anonymized after the review period.

## Overview

![Architecture overview](figures/overview_figure.png)

## Abstract

Many continual-learning methods modify the gradient upstream (projection,
penalty-based rescaling, replay-gradient mixing) and treat the adaptive
optimizer as a neutral backend. We exhibit a hidden failure mode of this
composition under Adam: in a high-overlap non-adaptive regime on an
8-domain continual language-model stream, every shared-routing projection
baseline collapses to within a fraction of a forgetting unit of vanilla,
and a naive fixed-strength decoupled variant drops below vanilla; only an
overlap-aware decoupled routing remains stable. We call the underlying
mechanism the attenuate-then-adapt conflict. Along a protected direction
the upstream modification attenuates the gradient and the classical
shared routing feeds the attenuated signal into both Adam moments; the
denominator then shrinks, implicitly inverting part of the intended
protection. A scalar-surrogate analysis in the projection home case
predicts a one-over-one-minus-alpha inflation of the old-direction
effective learning rate, matching measurement within eight percent across
eight alpha values. A denominator-only causal intervention and an
effective-learning-rate matching intervention together upgrade the
diagnosis from correlation to controlled intervention. The same conflict
recurs empirically in the penalty and replay families and across four
optimizer backbones. The repair, Adaptive-OGP, routes the attenuated
gradient into the first moment and the raw gradient into the second
moment, and modulates the attenuation strength with an overlap signal.

## Key findings

- Failure. Under a high-overlap non-adaptive regime, every shared-routing
  projection baseline (OGD, GPM, SGP, ROGO, FOPNG, Adam-NSCL) collapses to
  within about 0.7 forgetting units of vanilla, and a small replay buffer
  still trails Adaptive-OGP by roughly 2.2 units (Table 5).
- Diagnosis. The attenuate-then-adapt conflict is a property of
  composition, not of the individual module. A scalar surrogate predicts
  the old-direction effective learning-rate inflation within eight
  percent (Figure 1, Table 3), and matching the effective learning rate
  across routings closes about eighty percent of the forgetting gap
  under controlled intervention (Table 4).
- Repair. Adaptive-OGP attenuates only the numerator and leaves the
  denominator magnitude-faithful, and modulates the attenuation strength
  with an overlap signal; it remains stable across benchmarks, families,
  and optimizer backbones (Tables 1, 5, 6, 7).

## Installation

```
pip install -e .
```

Python 3.9 or newer is required. The default installation is CPU-only;
GPU wheels for PyTorch are optional and should be installed separately
if you plan to run the full-scale experiments.

## Hello-world demo

```
bash scripts/run_demo.sh
```

This runs every experiment in ``experiments/`` under its ``--demo`` mode.
The demo uses a tiny language model over a synthetic continual stream
and finishes in about a minute on CPU. Every run writes a structured
JSON record into ``runs/``; figures go to ``runs/*.png``.

## Quick sanity check

```
python -c "import adaptive_ogp; print(adaptive_ogp.__version__)"
pytest tests/
```

## Reproducing the main tables

Full-scale runs require an 8xA100 cluster (or equivalent) and the
8-domain / 16-domain / TRACE benchmarks. The demo mode is a structural
smoke test and is not a quantitative stand-in for the paper numbers.

| Paper artefact | Demo command | Full command |
| --- | --- | --- |
| Table 1 (clean 8-domain) | ``python experiments/exp01_clean_8domain.py --demo`` | ``python experiments/exp01_clean_8domain.py --config configs/8domain_256m.yaml`` |
| Table 2 (moment-pathway 2x2) | ``python experiments/exp02_moment_pathway_2x2.py --demo`` | ``python experiments/exp02_moment_pathway_2x2.py --config configs/8domain_256m.yaml`` |
| Table 3 (denominator intervention) | ``python experiments/exp03_denominator_intervention.py --demo`` | ``python experiments/exp03_denominator_intervention.py --config configs/8domain_256m.yaml`` |
| Table 4 (eta-matching intervention) | ``python experiments/exp04_eta_matching.py --demo`` | ``python experiments/exp04_eta_matching.py --config configs/8domain_256m.yaml`` |
| Table 5 (stress-overlap regime) | ``python experiments/exp05_stress_overlap.py --demo`` | ``python experiments/exp05_stress_overlap.py --config configs/8domain_256m.yaml`` |
| Table 6 (routing x schedule) | ``python experiments/exp06_routing_x_schedule.py --demo`` | ``python experiments/exp06_routing_x_schedule.py --config configs/8domain_256m.yaml`` |
| Table 7 (cross-family breadth) | ``python experiments/exp07_breadth.py --demo`` | ``python experiments/exp07_breadth.py --config configs/8domain_256m.yaml`` |
| 16-domain severity (Appendix) | ``python experiments/exp08_16domain_severity.py --demo`` | ``python experiments/exp08_16domain_severity.py --config configs/16domain_256m.yaml`` |
| 7B LoRA TRACE (Appendix) | ``python experiments/exp09_7b_lora.py --demo`` | ``python experiments/exp09_7b_lora.py --config configs/7b_lora.yaml`` |

``scripts/run_all_tables.sh`` aggregates the full-scale commands into a
single driver. The 7B LoRA config (``configs/7b_lora.yaml``) assumes
8xA100 80GB; there is no QLoRA path.

## Regenerating figures

```
python figures/fig2_vt_eta_evolution.py --out runs/fig2.png
python figures/fig3_overlap_realism.py  --out runs/fig3.png
python figures/fig4_penalty_mechanism.py --out runs/fig4.png
```

## Using Adaptive-OGP in your own code

```
import torch
from adaptive_ogp import AdaptiveOGP, OverlapAwareSchedule, SubspaceManager

model = ...                                   # any torch.nn.Module
subspace = SubspaceManager(rank=32)
schedule = OverlapAwareSchedule(alpha_max=0.5)

opt = AdaptiveOGP(
    model.parameters(),
    lr=1e-4,
    routing="ogp",
    subspace=subspace,
    schedule=schedule,
)

# --- train task A; collect gradients for the basis ---
for batch in task_a_loader:
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            opt.collect(p, p.grad.detach())
    opt.step()

opt.on_task_switch()       # materialises the protected subspace

# --- train task B; Adaptive-OGP applies ---
for batch in task_b_loader:
    opt.zero_grad(set_to_none=True)
    loss_fn(model, batch).backward()
    opt.step()
```

The routing flag takes values in ``{"vanilla", "v_only", "shared", "ogp",
"reverse"}`` so that the Table 2 ablation can be reproduced with a CLI
switch.

## Repository structure

```
attenuate-then-adapt/
  adaptive_ogp/           core library
    optimizer.py          AdaptiveOGP torch.optim.Optimizer
    routing.py            moment-pathway routing primitives
    schedule.py           overlap-aware adaptive strength
    subspace.py           SVD basis manager
    probes.py             v_t and R_eta diagnostic probes
    interventions.py      denominator-only and eta-matching controls
  baselines/              projection / penalty / replay baselines
  benchmarks/             8-domain, 16-domain, TRACE loaders / stubs
  experiments/            one driver per paper table
  figures/                figure scripts
  utils/                  metrics, seeding, logging
  configs/                YAML configs for full-scale runs
  scripts/                demo, full-sweep, download
  tests/                  pytest unit tests
```

## Design notes

- The core asymmetry ``attenuated gradient into m_t, raw gradient into
  v_t`` is implemented in ``adaptive_ogp/optimizer.py`` and is annotated
  with a pointer to the scalar-surrogate Proposition. The routing enum
  lives in ``adaptive_ogp/routing.py``.
- The overlap-aware schedule ``alpha_t = alpha_max * (1 - bar_s_t)`` is
  in ``adaptive_ogp/schedule.py`` with monotonicity tests in
  ``tests/test_schedule.py``.
- Subspace bases are extracted via randomized SVD (``torch.svd_lowrank``)
  from a bounded gradient buffer; see ``adaptive_ogp/subspace.py``.
- All experiment scripts fail fast on NaN gradients, out-of-range
  hyperparameters, and missing data. We do not silently fall back to
  defaults; errors are surfaced immediately.

## Citation

```
@inproceedings{anonymous2026adaptiveogp,
  title  = {Hidden Failure Modes of Gradient Modification under Adam in
            Continual Learning, and Adaptive Decoupled Moment Routing
            as a Repair},
  author = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year   = {2026},
  note   = {Under double-blind review.}
}
```

A de-anonymized citation will be provided after the review period.

## License

MIT, Copyright (c) 2026 Anonymous Authors. See ``LICENSE``.
