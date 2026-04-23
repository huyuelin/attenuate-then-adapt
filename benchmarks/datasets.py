"""Dataset download and preprocessing helpers.

This module is intentionally small. It does *not* bundle any dataset,
and it does *not* automate downloads in Python (downloads go through
``scripts/download_benchmarks.sh`` so that a reviewer can see exactly
what gets fetched). The only thing exposed here is a manifest of what
the 8-domain stream expects on disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    url: str
    sha256_prefix: str  # first 12 hex chars of expected archive SHA-256
    notes: str


MANIFEST_8DOMAIN: List[DatasetManifest] = [
    DatasetManifest(
        name="arxiv",
        url="https://path/to/arxiv-subset.tar.gz",
        sha256_prefix="000000000000",
        notes="Scientific abstracts subset used in Table 1.",
    ),
    DatasetManifest(
        name="book",
        url="https://path/to/book-subset.tar.gz",
        sha256_prefix="000000000000",
        notes="Narrative text.",
    ),
    DatasetManifest(
        name="dialogue",
        url="https://path/to/dialogue.tar.gz",
        sha256_prefix="000000000000",
        notes="Conversational text.",
    ),
    DatasetManifest(
        name="gutenberg",
        url="https://path/to/gutenberg.tar.gz",
        sha256_prefix="000000000000",
        notes="Project Gutenberg subset.",
    ),
    DatasetManifest(
        name="medical",
        url="https://path/to/medical.tar.gz",
        sha256_prefix="000000000000",
        notes="Medical abstracts subset.",
    ),
    DatasetManifest(
        name="news",
        url="https://path/to/news.tar.gz",
        sha256_prefix="000000000000",
        notes="News article subset.",
    ),
    DatasetManifest(
        name="openweb",
        url="https://path/to/openweb.tar.gz",
        sha256_prefix="000000000000",
        notes="OpenWebText slice.",
    ),
    DatasetManifest(
        name="code",
        url="https://path/to/code.tar.gz",
        sha256_prefix="000000000000",
        notes="Permissively-licensed code subset.",
    ),
]


def manifest_lines() -> List[str]:
    """Return a human-readable manifest for printing from the CLI."""
    return [
        f"{m.name:<10} {m.url}  [sha256 {m.sha256_prefix}...]  {m.notes}"
        for m in MANIFEST_8DOMAIN
    ]
