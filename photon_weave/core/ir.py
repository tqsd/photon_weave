"""
Lightweight, serializable circuit specs used by the runtime executor.

These dataclasses are pure data: they do not depend on envelopes or state
classes and are intended to be JSON/msgpack friendly. Operators are kept as
arrays so existing kernels can be reused without additional builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple


@dataclass(frozen=True)
class StateSpec:
    """
    Description of a quantum state tensor.

    Attributes
    ----------
    dims : tuple[int, ...]
        Per-subsystem dimensions in tensor order.
    rep : str
        Representation, e.g., ``\"vector\"`` or ``\"matrix\"``.
    data : Any | None
        Optional tensor payload; when None, the executor expects a tensor to be
        supplied explicitly.
    """

    dims: Tuple[int, ...]
    rep: str
    data: Any | None = None


@dataclass(frozen=True)
class OpSpec:
    """
    Operation applied to one or more targets.

    Attributes
    ----------
    name : str
        Human-readable operation name.
    operator : Any
        Operator tensor matching the representation (vector: [d, d];
        matrix: [d, d]).
    targets : tuple[int, ...]
        Target subsystem indices (0-based).
    rep : str
        Representation the operator expects, ``\"vector\"`` or ``\"matrix\"``.
    use_contraction : bool | None
        Whether to route through the contraction path; when None, executor
        defaults to global Config.
    """

    name: str
    operator: Any
    targets: Tuple[int, ...]
    rep: str = "vector"
    use_contraction: bool | None = None


@dataclass(frozen=True)
class MeasureSpec:
    """
    Measurement request for one or more targets.

    Attributes
    ----------
    targets : tuple[int, ...]
        Target subsystem indices (0-based).
    rep : str
        Representation to measure in, ``\"vector\"`` or ``\"matrix\"``.
    povm : Any | None
        Optional POVM operators (matrix representation only).
    """

    targets: Tuple[int, ...]
    rep: str = "vector"
    povm: Any | None = None


@dataclass(frozen=True)
class TraceOutSpec:
    """
    Trace out everything except the given targets.
    """

    targets: Tuple[int, ...]


@dataclass(frozen=True)
class CircuitSpec:
    """
    Minimal circuit description.

    Attributes
    ----------
    state : StateSpec
        Initial state description (dims + tensor).
    steps : tuple[object, ...]
        Sequence of OpSpec, MeasureSpec, or TraceOutSpec instances.
    """

    state: StateSpec
    steps: Tuple[object, ...]

    def with_steps(self, steps: Iterable[object]) -> "CircuitSpec":
        return CircuitSpec(self.state, tuple(steps))
