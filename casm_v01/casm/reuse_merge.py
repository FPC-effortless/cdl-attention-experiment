from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ReuseSignature:
    entity_id: int
    version: int
    dependency_signature: tuple[int, ...]


def compatible(a: ReuseSignature, b: ReuseSignature) -> bool:
    return (
        a.entity_id == b.entity_id
        and a.version == b.version
        and a.dependency_signature == b.dependency_signature
    )


def propose_reuse_groups(signatures: Iterable[ReuseSignature]) -> tuple[tuple[int, ...], ...]:
    groups: list[list[int]] = []
    reps: list[ReuseSignature] = []
    for idx, sig in enumerate(signatures):
        for group_idx, rep in enumerate(reps):
            if compatible(sig, rep):
                groups[group_idx].append(idx)
                break
        else:
            reps.append(sig)
            groups.append([idx])
    return tuple(tuple(group) for group in groups)
