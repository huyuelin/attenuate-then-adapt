"""Metrics used across continual-learning tables in the paper.

* ``forgetting`` (Table 1, 5, 6, 7): averaged drop in old-task metric
  between the end-of-its-own-training checkpoint and the end of the full
  stream.
* ``accuracy``: standard classification accuracy.
* ``adapt_at_k`` (``adapt@500`` in the paper): new-task metric measured
  after ``k`` optimizer steps on the new task.
* ``perplexity``: exp of the cross-entropy loss.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 classification accuracy in [0,1]."""
    assert logits.ndim >= 2, "logits must be at least 2-D"
    assert targets.ndim == logits.ndim - 1, "targets must be one dim fewer than logits"
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float().mean().item()
    assert math.isfinite(correct), "non-finite accuracy"
    return correct


def perplexity(loss: float | torch.Tensor) -> float:
    """Convert mean cross-entropy loss to perplexity."""
    val = float(loss.item() if torch.is_tensor(loss) else loss)
    assert math.isfinite(val) and val >= 0.0, f"bad loss: {val}"
    return math.exp(val)


def forgetting(per_task_history: Sequence[Sequence[float]]) -> float:
    """Average forgetting over all tasks except the last.

    Parameters
    ----------
    per_task_history: sequence of sequences.
        ``per_task_history[i][j]`` = metric on task ``i`` measured after
        finishing task ``j`` (``j >= i``). Higher-is-better metric.

    Returns
    -------
    float
        ``mean_i (max_j history[i][j] - history[i][-1])`` over
        i = 0 .. len - 2.
    """
    assert len(per_task_history) >= 2, "need at least two tasks to compute forgetting"
    vals = []
    for i, hist in enumerate(per_task_history[:-1]):
        assert len(hist) >= 1, f"task {i} has empty history"
        best = max(hist)
        final = hist[-1]
        vals.append(best - final)
    return sum(vals) / len(vals)


def adapt_at_k(
    per_step_metric: Iterable[float],
    k: int,
) -> float:
    """Metric at step k on the new task (zero-indexed).

    Fast-fails if the iterable is shorter than k + 1 entries; this avoids
    silently reporting the last available value and pretending it was
    measured at step k.
    """
    assert k >= 0, f"k must be non-negative, got {k}"
    seq = list(per_step_metric)
    assert len(seq) > k, f"history length {len(seq)} <= k={k}"
    return float(seq[k])
