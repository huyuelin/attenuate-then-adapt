"""Utility helpers: deterministic seeding, metrics, structured logging."""
from utils.metrics import accuracy, adapt_at_k, forgetting, perplexity
from utils.seed import set_deterministic_seed
from utils.logging import JSONLogger

__all__ = [
    "accuracy",
    "adapt_at_k",
    "forgetting",
    "perplexity",
    "set_deterministic_seed",
    "JSONLogger",
]
