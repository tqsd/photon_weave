# Update Notes

This page summarizes the recent changes and briefly explains the new core modules, metadata helpers, and shape planning utilities.

## Highlights
- Expanded measurement docstrings (vector/matrix, JIT wrappers, POVM/PNR, expectations) with explicit shapes, key handling, meta routing, and noise models.
- Added core adapter/kernels modules with Kraus-aware contraction paths and jitted entry points.
- Cleaned documentation structure (headings, code blocks, myst support) and added API/core overview pages.

## Core Modules
- `photon_weave.core.adapters`: Reorder/reshape flat tensors so targets sit at the front, call ordered kernels, optionally use contraction (opt_einsum), and restore original ordering. Provides `*_jit` and `*_jit_meta` for static-shape JIT paths.
- `photon_weave.core.kernels`: Stateless, ordered tensor primitives (assume targets already front-most). Implements apply/measure/trace for vectors and matrices plus Kraus and POVM paths.
- `photon_weave.core.linear` and `photon_weave.core.ops`: Thin dispatch layers to adapters/kernels; keep array-only interfaces.
- `photon_weave.core.meta`: Metadata helpers (dims, target indices) to avoid recomputation and keep shapes static for JIT/vmap.
- `photon_weave.core.rng`: Key handling utilities (borrow/split) used by measurement paths.
- `photon_weave.core.jitted`: Convenience wrappers that bind metas for JIT-friendly call sites.

## Meta and Shape Planning
- `photon_weave.core.meta.DimsMeta`: Captures static dimensions, target indices, and rest indices for a given operation/measurement. Enables reuse and stable shapes under JIT.
- `photon_weave.state.utils.shape_planning.ShapePlan`: Higher-level plan pairing `DimsMeta` with source/target ordering for state objects. Used by compiled kernels to skip per-call shape computation.
- Shape planning flow: build `ShapePlan` (or `DimsMeta`) once, route operations/measurements through `*_jit_meta` adapters/jitted helpers to avoid repeated reshapes/index lookups and to maintain static shapes for compilation.

## Measurement Updates
- RNG splitting now drops the first subkey to decorrelate runs when different base keys are provided.
- POVM/Kraus paths validate operator shapes; envelope post-measure/Kraus paths contract once (and twice when contractions are enabled) to restore standalone representations.
- PNR helpers model efficiency (binomial thinning), dark counts (Poisson), and jitter (Gaussian) with both vector and matrix support.
