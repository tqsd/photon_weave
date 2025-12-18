"""
Placeholder adapters for Piquasso ↔ Weave Circuits.

These functions should map Piquasso programs/states to the `CircuitSpec`
representation and back.
"""

from __future__ import annotations

from photon_weave.core.ir import CircuitSpec


def from_piquasso(obj: object) -> CircuitSpec:
    raise NotImplementedError(
        "Piquasso interop not yet implemented; returns CircuitSpec"
    )


def to_piquasso(spec: CircuitSpec) -> object:
    raise NotImplementedError(
        "Piquasso interop not yet implemented; accepts CircuitSpec"
    )
