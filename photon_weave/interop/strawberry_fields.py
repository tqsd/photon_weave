"""
Placeholder adapters for Strawberry Fields ↔ Weave Circuits.

These functions should map Strawberry Fields circuit/state objects to the
framework-agnostic `CircuitSpec` and back.
"""

from __future__ import annotations

from photon_weave.core.ir import CircuitSpec


def from_strawberry_fields(obj: object) -> CircuitSpec:
    raise NotImplementedError(
        "Strawberry Fields interop not yet implemented; returns CircuitSpec"
    )


def to_strawberry_fields(spec: CircuitSpec) -> object:
    raise NotImplementedError(
        "Strawberry Fields interop not yet implemented; accepts CircuitSpec"
    )
