"""
Metadata helpers for JIT-friendly core calls.

`DimsMeta` collects the dimension tuple and target/rest indices so they can be
passed as static arguments to jitted entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DimsMeta:
    state_dims: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    rest_indices: Tuple[int, ...]


def make_meta(state_dims: Tuple[int, ...], target_indices: Tuple[int, ...]) -> DimsMeta:
    rest_indices = tuple(i for i in range(len(state_dims)) if i not in target_indices)
    return DimsMeta(
        state_dims=tuple(state_dims),
        target_indices=tuple(target_indices),
        rest_indices=rest_indices,
    )
