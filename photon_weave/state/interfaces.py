"""
Lightweight protocol-style interfaces to avoid runtime import cycles.

These are intentionally minimal and only describe the attributes/methods that
the utility helpers require for typing; they do not depend on the concrete
state classes.
"""

from __future__ import annotations

from typing import Protocol, Tuple, Union

import jax.numpy as jnp


class BaseStateLike(Protocol):
    index: Union[int, Tuple[int, int], None]
    dimensions: int
    expansion_level: object
    envelope: object | None
    composite_envelope: object | None
    state: object
    _num_quanta: int | None

    def expand(self) -> None: ...
    def _set_measured(self) -> None: ...
    def resize(self, new_dimensions: int) -> bool: ...
    def trace_out(self): ...


class EnvelopeLike(Protocol):
    fock: BaseStateLike
    polarization: BaseStateLike
    state: jnp.ndarray | None
    expansion_level: object


class CompositeEnvelopeLike(Protocol):
    product_states: list


class OperationLike(Protocol):
    _operation_type: object
    dimensions: list
    operator: object
    renormalize: bool

    def compute_dimensions(self, num_quanta, state, *args, **kwargs): ...
