"""Experiment drivers.

Each ``exp0X_*.py`` reproduces one table (or one figure supporting
table) in the paper. Every driver supports ``--demo`` for a short CPU
run and writes a structured JSON record so that downstream aggregation
is deterministic.

``exp01_clean_8domain.py``        -> Table 1
``exp02_moment_pathway_2x2.py``   -> Table 2
``exp03_denominator_intervention.py`` -> Table 3
``exp04_eta_matching.py``         -> Table 4
``exp05_stress_overlap.py``       -> Table 5
``exp06_routing_x_schedule.py``   -> Table 6
``exp07_breadth.py``              -> Table 7
``exp08_16domain_severity.py``    -> Appendix long-sequence table
``exp09_7b_lora.py``              -> 7B LoRA TRACE (stub)
"""
