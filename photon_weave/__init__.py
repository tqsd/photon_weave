"""Top-level PhotonWeave helpers."""

# Keep JAX RNG behavior stable across versions by pinning the legacy PRNG
# implementation. This needs to run before importing modules that draw keys.

import jax

from photon_weave import _math, constants, core, extra, operation, state
from photon_weave.state.utils.shape_planning import (
    ShapePlan,
    build_plan,
    compiled_kernels,
    make_meta,
)

jax.config.update("jax_default_prng_impl", "threefry2x32")


__all__ = [
    "constants",
    "core",
    "extra",
    "_math",
    "operation",
    "state",
    "build_plan",
    "compiled_kernels",
    "make_meta",
    "ShapePlan",
]
