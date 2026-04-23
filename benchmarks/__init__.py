"""Benchmark loaders used in the paper.

* ``continual_lm_8domain``: 8-domain continual-LM stream (Section 5.1,
  Table 1 and Table 5). Provides a toy-mode on-the-fly generator for
  fast reproduction of the structural effect on CPU, plus a pointer to
  the real-data download path for full-scale runs.
* ``continual_lm_16domain``: long-sequence extension (Appendix Table).
* ``trace_benchmark``: TRACE stub (Yang et al., 2024) used in Section
  ``Cross-family breadth'' for the 7B LoRA experiment.
"""
from benchmarks.continual_lm_8domain import (
    EightDomainStream,
    build_toy_stream,
)
from benchmarks.trace_benchmark import TRACEBenchmarkStub

__all__ = [
    "EightDomainStream",
    "build_toy_stream",
    "TRACEBenchmarkStub",
]
