"""TRACE benchmark stub used by the 7B LoRA experiment.

The real TRACE benchmark (Yang et al., 2024) ships its own dataset
loaders. Because this reference implementation keeps the repository
under 2 MB and does not bundle datasets, this module exposes only a
schema-compatible stub that declares the task order, expected metrics,
and split boundaries. The 7B LoRA runner in ``experiments/exp09_7b_lora``
imports this stub and either resolves it against a local TRACE install
(if ``TRACE_ROOT`` is set) or raises a clear ``NotImplementedError``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TRACETask:
    name: str
    metric: str
    max_examples: int


TRACE_TASKS: List[TRACETask] = [
    TRACETask("CStance",   metric="accuracy", max_examples=2048),
    TRACETask("FOMC",      metric="accuracy", max_examples=2048),
    TRACETask("MeetingBank", metric="rouge-L", max_examples=2048),
    TRACETask("Py150",     metric="exact-match", max_examples=2048),
    TRACETask("ScienceQA", metric="accuracy", max_examples=2048),
    TRACETask("NumGLUE-cm", metric="accuracy", max_examples=2048),
    TRACETask("NumGLUE-ds", metric="accuracy", max_examples=2048),
    TRACETask("20Minuten", metric="rouge-L", max_examples=2048),
]


class TRACEBenchmarkStub:
    """Deferred loader for TRACE.

    Usage
    -----
    ``stub = TRACEBenchmarkStub()`` imports without data. Call
    ``stub.require_local()`` before iteration to either resolve the
    benchmark against ``$TRACE_ROOT`` or raise a clear error that tells
    the reviewer what to do (run ``scripts/download_benchmarks.sh``).
    """

    def __init__(self) -> None:
        self.root = os.environ.get("TRACE_ROOT")

    def require_local(self) -> None:
        if self.root is None or not os.path.isdir(self.root):
            raise NotImplementedError(
                "TRACE benchmark not installed locally; see scripts/run_7b_lora.sh "
                "and set TRACE_ROOT to the extracted benchmark directory. "
                "The repository deliberately does not bundle the benchmark."
            )

    def task_list(self) -> List[TRACETask]:
        return list(TRACE_TASKS)
