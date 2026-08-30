from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .benchmark_v1 import prompt_hash_from_training_text


def append_training_prompt_hashes(path: str | Path, example_texts: Iterable[str]) -> int:
    """Append visible-prefix hashes for generated training episodes.

    Call this on every training batch. The hash excludes gold answer bytes, so
    exact benchmark prompt overlap can be detected later even though training
    examples contain their answers.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hashes = [prompt_hash_from_training_text(text) for text in example_texts]
    with p.open("a", encoding="utf-8") as f:
        for h in hashes:
            f.write(h + "\n")
    return len(hashes)


def compact_training_prompt_hashes(path: str | Path) -> int:
    """Deduplicate/sort a completed prompt-hash log for reproducible artifacts."""
    p = Path(path)
    values = sorted({x.strip() for x in p.read_text().splitlines() if x.strip()})
    p.write_text("".join(x + "\n" for x in values), encoding="utf-8")
    return len(values)
