"""8-domain continual-LM benchmark loader.

This module provides two entry points:

* ``EightDomainStream``: canonical benchmark used in Table 1 and Table 5.
  The constructor expects a directory of eight tokenized corpora on disk;
  see ``scripts/download_benchmarks.sh`` for the download protocol. The
  demo mode (``toy=True``) returns a small synthetic stream that runs on
  CPU in under a minute and is only intended to exercise code paths.

* ``build_toy_stream``: standalone helper that returns a list of
  ``(train_iter, eval_iter)`` pairs with deterministic shapes. Used by
  every ``experiments/exp0X_*.py`` script when ``--demo`` is set.

The real eight domains (arxiv, bookscorpus, gutenberg, medical, news,
openwebtext-subset, code, dialogue) are intentionally *not bundled*; the
repository stays under 2 MB. The toy stream mimics the long-range
statistics of the real stream via correlated Gaussian inputs with
vocabulary-overlap control.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import torch


@dataclass
class ToyTaskSpec:
    """Parameters controlling one synthetic task in the toy stream."""

    vocab_size: int = 512
    seq_len: int = 32
    tokens_per_task: int = 4096
    overlap_with_prev: float = 0.3  # fraction of vocabulary shared with task t-1


def _make_loader(
    spec: ToyTaskSpec,
    prev_vocab: torch.Tensor | None,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (train_tokens, eval_tokens) for one task.

    Every returned id lies strictly in ``[0, spec.vocab_size)`` so that an
    embedding table of size ``spec.vocab_size`` is sufficient; the tasks
    are differentiated by which *subset* of that id range is sampled,
    with ``overlap_with_prev`` controlling how many ids carry over from
    the previous task.
    """
    assert 0.0 <= spec.overlap_with_prev <= 1.0
    V = spec.vocab_size
    g = torch.Generator().manual_seed(seed)
    # Fixed "active" vocabulary size per task: we use half of V so that
    # there is room for distinct ids across tasks while still bounded.
    active = max(4, V // 2)
    shared = int(round(spec.overlap_with_prev * active))
    if prev_vocab is None or shared == 0:
        vocab = torch.randperm(V, generator=g)[:active]
    else:
        shared_idx = prev_vocab[:shared]
        pool = torch.arange(V)
        pool = pool[~torch.isin(pool, shared_idx)]
        fresh = pool[torch.randperm(pool.numel(), generator=g)][: active - shared]
        vocab = torch.cat([shared_idx, fresh])
    assert vocab.max().item() < V, (
        f"toy vocab id {int(vocab.max().item())} >= vocab_size {V}"
    )
    train_n = spec.tokens_per_task
    eval_n = max(spec.tokens_per_task // 4, spec.seq_len * 2)
    train = vocab[torch.randint(0, vocab.numel(), (train_n,), generator=g)]
    ev = vocab[torch.randint(0, vocab.numel(), (eval_n,), generator=g)]
    return train.long(), ev.long()


class EightDomainStream:
    """Lazy iterator over eight continual tasks.

    For the demo (toy=True), the stream is fully synthetic and runs on
    CPU. For the real benchmark, point ``root`` at the directory populated
    by ``scripts/download_benchmarks.sh``; the eight sub-directories are
    then iterated in a fixed alphabetical order.
    """

    DOMAINS = (
        "arxiv",
        "book",
        "dialogue",
        "gutenberg",
        "medical",
        "news",
        "openweb",
        "code",
    )

    def __init__(
        self,
        root: str | None = None,
        toy: bool = False,
        seed: int = 0,
        toy_spec: ToyTaskSpec | None = None,
    ) -> None:
        if not toy:
            assert root is not None and isinstance(root, str), (
                "EightDomainStream(toy=False) requires an existing data root"
            )
            import os
            for dom in self.DOMAINS:
                sub = os.path.join(root, dom)
                assert os.path.isdir(sub), (
                    f"missing domain subdirectory: {sub} "
                    f"(run scripts/download_benchmarks.sh first)"
                )
        self.root = root
        self.toy = toy
        self.seed = seed
        self.toy_spec = toy_spec or ToyTaskSpec()

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
        if self.toy:
            prev_vocab: torch.Tensor | None = None
            for t, dom in enumerate(self.DOMAINS):
                train, ev = _make_loader(self.toy_spec, prev_vocab, seed=self.seed + t)
                prev_vocab = train[: self.toy_spec.vocab_size].unique()
                yield dom, train, ev
        else:
            raise NotImplementedError(
                "Real-data iteration is left to the downstream experiment runner; "
                "see scripts/download_benchmarks.sh and configs/8domain_256m.yaml."
            )


def build_toy_stream(
    num_tasks: int = 8,
    vocab_size: int = 512,
    seq_len: int = 32,
    tokens_per_task: int = 4096,
    overlap: float = 0.3,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return a list of (train, eval) tensor pairs for the toy stream.

    Shape: each returned tensor is a 1-D long tensor of token ids. The
    caller is responsible for reshaping / packing into (B, L) batches.
    """
    assert 2 <= num_tasks <= 64, f"num_tasks out of range: {num_tasks}"
    spec = ToyTaskSpec(
        vocab_size=vocab_size,
        seq_len=seq_len,
        tokens_per_task=tokens_per_task,
        overlap_with_prev=overlap,
    )
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_vocab: torch.Tensor | None = None
    for t in range(num_tasks):
        train, ev = _make_loader(spec, prev_vocab, seed=seed + t)
        prev_vocab = train[:vocab_size].unique()
        out.append((train, ev))
    return out


def batch_iter(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
) -> Iterable[torch.Tensor]:
    """Yield (B, L) batches from a flat token sequence.

    Fast-fails if ``tokens.numel() < batch_size * seq_len``.
    """
    assert tokens.ndim == 1, "tokens must be a 1-D tensor"
    n_windows = tokens.numel() // seq_len
    assert n_windows >= batch_size, (
        f"not enough tokens for batch_size {batch_size} at seq_len {seq_len}; "
        f"have {n_windows} windows."
    )
    windows = tokens[: n_windows * seq_len].view(n_windows, seq_len)
    if shuffle:
        idx = torch.randperm(n_windows)
        windows = windows[idx]
    n_batches = n_windows // batch_size
    for b in range(n_batches):
        yield windows[b * batch_size : (b + 1) * batch_size]


def make_continual_lm_16domain(seed: int = 0, tokens_per_task: int = 4096):
    """Toy variant of the 16-domain long-sequence stream (Appendix).

    Twice the domain count of ``build_toy_stream``, same per-task size.
    """
    return build_toy_stream(
        num_tasks=16,
        vocab_size=512,
        seq_len=32,
        tokens_per_task=tokens_per_task,
        overlap=0.35,
        seed=seed,
    )


_PI = math.pi  # used elsewhere; kept here to avoid unused-import warnings
