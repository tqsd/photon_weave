"""
Placeholder adapters for Qiskit ↔ Weave Circuits.

These functions should map Qiskit circuits/states to the `CircuitSpec`
representation and back.
"""

from __future__ import annotations

from photon_weave.core.ir import CircuitSpec


def from_qiskit(obj: object) -> CircuitSpec:
    raise NotImplementedError("Qiskit interop not yet implemented; returns CircuitSpec")


def to_qiskit(spec: CircuitSpec) -> object:
    raise NotImplementedError("Qiskit interop not yet implemented; accepts CircuitSpec")
