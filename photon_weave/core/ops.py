"""
Jittable operator helpers that wrap the legacy `_math.ops` definitions.
"""

from __future__ import annotations

from functools import reduce
from typing import Sequence

import jax
import jax.numpy as jnp

from photon_weave._math import ops as legacy_ops

# Stateless wrappers; legacy ops are already jitted but we re-expose from core.
identity_operator = legacy_ops.identity_operator
hadamard_operator = legacy_ops.hadamard_operator
x_operator = legacy_ops.x_operator
y_operator = legacy_ops.y_operator
z_operator = legacy_ops.z_operator
s_operator = legacy_ops.s_operator
t_operator = legacy_ops.t_operator
controlled_not_operator = legacy_ops.controlled_not_operator
controlled_z_operator = legacy_ops.controlled_z_operator
swap_operator = legacy_ops.swap_operator
sx_operator = legacy_ops.sx_operator
controlled_swap_operator = legacy_ops.controlled_swap_operator
rx_operator = legacy_ops.rx_operator
ry_operator = legacy_ops.ry_operator
rz_operator = legacy_ops.rz_operator
displacement_operator = legacy_ops.displacement_operator
phase_operator = legacy_ops.phase_operator
squeezing_operator = legacy_ops.squeezing_operator
creation_operator = legacy_ops.creation_operator
annihilation_operator = legacy_ops.annihilation_operator
apply_kraus = legacy_ops.apply_kraus
kraus_identity_check = legacy_ops.kraus_identity_check
num_quanta_vector = legacy_ops.num_quanta_vector
num_quanta_matrix = legacy_ops.num_quanta_matrix
jitted_exp = legacy_ops.jitted_exp
u3_operator = legacy_ops.u3_operator
compute_einsum = legacy_ops.compute_einsum


@jax.jit
def fidelity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Simple fidelity helper for two state vectors.
    """
    return jnp.abs(jnp.vdot(a, b)) ** 2


@jax.jit
def kron_reduce(arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """
    Jittable Kronecker product over a sequence of matrices.

    If `arrays` is empty, returns a 1x1 identity. Uses `jnp.kron` under the hood
    and a left fold to minimize intermediate allocations.
    """
    if not arrays:
        return jnp.array([[1]])
    init = arrays[0]
    return reduce(lambda a, b: jnp.kron(a, b), arrays[1:], init)


__all__ = [
    "identity_operator",
    "hadamard_operator",
    "x_operator",
    "y_operator",
    "z_operator",
    "s_operator",
    "t_operator",
    "controlled_not_operator",
    "controlled_z_operator",
    "swap_operator",
    "sx_operator",
    "controlled_swap_operator",
    "rx_operator",
    "ry_operator",
    "rz_operator",
    "jitted_exp",
    "fidelity",
    "kron_reduce",
    "compute_einsum",
]
