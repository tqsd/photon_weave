"""
Core tensor kernels and adapters for JIT/vmap-friendly simulation steps.

These functions operate purely on arrays plus lightweight metadata
(dimensions, target counts) and avoid any envelope/composite state.
"""

from photon_weave.core import adapters, jitted, kernels, meta, ops, rng

__all__ = ["kernels", "adapters", "jitted", "meta", "rng", "ops"]
