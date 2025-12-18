"""
Interop stubs for framework bridges targeting the IR layer.

These helpers will translate to/from `CircuitSpec` without pulling in heavy
dependencies at import time. Implementations remain to be added.
"""

from .piquasso import from_piquasso, to_piquasso
from .qiskit import from_qiskit, to_qiskit
from .strawberry_fields import from_strawberry_fields, to_strawberry_fields

__all__ = [
    "from_piquasso",
    "to_piquasso",
    "from_qiskit",
    "to_qiskit",
    "from_strawberry_fields",
    "to_strawberry_fields",
]
