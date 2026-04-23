"""Shared toy-model and trainer used by ``experiments/exp0X_*.py --demo``.

The demo model is a two-layer token-classification head over an
embedding table: it has well-defined 2-D weight matrices so that the
SVD subspace manager has something to project onto, and it is small
enough to train on CPU in under a minute. For the full-scale 256M HOPE
runs, the same driver interface is consumed by an external trainer
configured in ``configs/8domain_256m.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_ogp.routing import RoutingMode
from adaptive_ogp.subspace import SubspaceManager
from benchmarks.continual_lm_8domain import batch_iter, build_toy_stream


class ToyLanguageModel(nn.Module):
    """Tiny LM for demo runs: embed -> MLP -> tied-softmax output.

    Total parameter count on default settings is under 200k.
    """

    def __init__(self, vocab_size: int = 512, d_model: int = 64, d_hidden: int = 128):
        super().__init__()
        assert vocab_size > 1, f"bad vocab_size {vocab_size}"
        assert d_model > 0 and d_hidden > 0
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=True)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights to keep the model small.
        self.out.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) integer tokens
        h = self.embed(x)          # (B, L, D)
        h = F.gelu(self.fc1(h))    # (B, L, H)
        h = self.fc2(h)            # (B, L, D)
        return self.out(h)          # (B, L, V)


def cross_entropy_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Shift-by-one next-token loss on a (B, L) token batch.

    The loss is masked so that padding tokens (id 0) do not contribute.
    """
    assert batch.ndim == 2, f"expected (B, L), got shape {tuple(batch.shape)}"
    logits = model(batch[:, :-1])
    targets = batch[:, 1:].contiguous()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,  # no ignore; keep signature simple for testing
    )
    return loss


@dataclass
class TrainerOutputs:
    per_task_eval_history: List[List[float]]  # shape (num_tasks, num_tasks)
    first_task_adapt_curve: List[float]        # new-task loss vs step on task 1
    final_perplexity: float


def _evaluate(model: nn.Module, tokens: torch.Tensor, batch_size: int, seq_len: int) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in batch_iter(tokens, batch_size, seq_len, shuffle=False):
            losses.append(float(cross_entropy_loss(model, batch).item()))
    model.train()
    assert losses, "eval iterator produced no batches; check seq_len/batch_size"
    return sum(losses) / len(losses)


def run_continual_demo(
    optimizer_factory: Callable[[nn.Module, Optional[SubspaceManager]], torch.optim.Optimizer],
    num_tasks: int = 8,
    tokens_per_task: int = 4096,
    seq_len: int = 32,
    batch_size: int = 16,
    overlap: float = 0.3,
    seed: int = 0,
    subspace_rank: int = 8,
    collect_window: int = 32,
    epochs: int = 1,
    routing_needs_subspace: bool = True,
) -> TrainerOutputs:
    """Run a continual demo loop and report per-task evaluation history.

    Parameters
    ----------
    optimizer_factory : callable
        Returns an optimizer bound to the model's parameters. The second
        argument is the ``SubspaceManager`` (may be ``None``).
    """
    assert num_tasks >= 2, "need at least two tasks for continual learning"
    torch.manual_seed(seed)

    model = ToyLanguageModel()
    subspace: Optional[SubspaceManager] = None
    if routing_needs_subspace:
        subspace = SubspaceManager(rank=subspace_rank, buffer_capacity=subspace_rank * 4)
    opt = optimizer_factory(model, subspace)

    stream = build_toy_stream(
        num_tasks=num_tasks,
        tokens_per_task=tokens_per_task,
        seq_len=seq_len,
        overlap=overlap,
        seed=seed,
    )

    # History[i][j] = loss on task i evaluated after finishing task j.
    history: List[List[float]] = [[] for _ in range(num_tasks)]
    first_task_curve: List[float] = []

    for task_idx, (train_tokens, _) in enumerate(stream):
        # During the final epoch of each task, collect gradients into
        # the subspace manager so that the *next* task starts with a
        # fresh protected basis.
        for epoch in range(epochs):
            for step, batch in enumerate(batch_iter(train_tokens, batch_size, seq_len)):
                opt.zero_grad(set_to_none=True)
                loss = cross_entropy_loss(model, batch)
                assert torch.isfinite(loss), f"NaN loss at task {task_idx} step {step}"
                loss.backward()

                # Collect a small window of gradients at the very end of
                # the task so that a rank-r basis is available at the
                # next boundary.
                if epoch == epochs - 1 and subspace is not None:
                    if step >= (tokens_per_task // (batch_size * seq_len)) - collect_window:
                        for p in model.parameters():
                            if p.grad is not None:
                                subspace.collect(p, p.grad.detach())
                opt.step()

                if task_idx == 1:
                    first_task_curve.append(float(loss.item()))

        if subspace is not None and task_idx < num_tasks - 1:
            subspace.build_bases()
            if hasattr(opt, "on_task_switch"):
                opt.on_task_switch()

        # Evaluate on every task seen so far.
        for eval_idx, (_, eval_tokens) in enumerate(stream[: task_idx + 1]):
            history[eval_idx].append(_evaluate(model, eval_tokens, batch_size, seq_len))

    final_loss = history[-1][-1]
    return TrainerOutputs(
        per_task_eval_history=history,
        first_task_adapt_curve=first_task_curve,
        final_perplexity=float(torch.tensor(final_loss).exp().item()),
    )


def factory_adam(model: nn.Module, _subspace: Optional[SubspaceManager]) -> torch.optim.Optimizer:
    """Return a plain Adam optimizer (shared-routing vanilla baseline)."""
    return torch.optim.Adam(model.parameters(), lr=1.0e-3)


def factory_adaptive_ogp(
    model: nn.Module,
    subspace: Optional[SubspaceManager],
    alpha_max: float = 0.5,
    routing: RoutingMode = RoutingMode.OGP,
) -> torch.optim.Optimizer:
    """Return an ``AdaptiveOGP`` optimizer with an overlap-aware schedule."""
    assert subspace is not None, "Adaptive-OGP requires a SubspaceManager"
    from adaptive_ogp.optimizer import AdaptiveOGP
    from adaptive_ogp.schedule import OverlapAwareSchedule

    schedule = OverlapAwareSchedule(alpha_max=alpha_max)
    return AdaptiveOGP(
        model.parameters(),
        lr=1.0e-3,
        routing=routing,
        subspace=subspace,
        schedule=schedule,
        alpha_max=alpha_max,
    )


def forgetting_from_history(history: List[List[float]]) -> float:
    """Same convention as ``utils.metrics.forgetting`` but on *loss*
    (lower-is-better), so we negate before/after reporting.

    Returns the mean final-minus-best loss gap over tasks 0..T-2, where
    ``best`` is the *lowest* loss achieved at any point after training
    that task. Higher values indicate more forgetting.
    """
    assert len(history) >= 2, "need >= 2 tasks"
    vals = []
    for i, h in enumerate(history[:-1]):
        assert len(h) >= 1, f"task {i} has empty history"
        best = min(h)
        final = h[-1]
        vals.append(final - best)
    return sum(vals) / len(vals)
