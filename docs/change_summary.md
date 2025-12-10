# Change Summary and Core Architecture Notes

This note captures the recent changes and clarifies how the core module, metadata helpers, and shape planning fit together. It is meant to be a quick orientation for contributors who want the rationale behind the latest updates.

## Recent Changes
- Expanded measurement docstrings (vector/matrix, JIT wrappers, POVM/PNR, expectations) with shapes, key handling, noise models, and meta routing described explicitly.
- Improved Kraus application paths to validate operator shapes, use consistent contractions, and restore standalone state ordering after applying operators.
- Documented adapters/kernels interactions, including JIT entry points and how contraction flags affect execution.
- Enabled Markdown support in docs via `myst_parser` and cleaned heading/code-block formatting across the docs set.
- Added tests for adapters, jitted paths, metadata planning, RNG decorrelation, and measurement utilities to guard the new behavior.

## Core Module Overview
- `photon_weave.core.adapters`: Reshapes/reorders tensors so targets lead the axis order, invokes ordered kernels, applies optional opt_einsum contraction, then restores the original layout. Provides `*_jit` and `*_jit_meta` helpers for static-shape call sites.
- `photon_weave.core.kernels`: Stateless tensor primitives that assume targets are already front-most; implement apply/measure/trace for vectors and matrices plus Kraus and POVM paths.
- `photon_weave.core.linear` and `photon_weave.core.ops`: Thin dispatch layers that expose array-focused APIs and forward to adapters/kernels.
- `photon_weave.core.jitted`: Convenience wrappers that bind metadata for JIT-friendly invocation.
- `photon_weave.core.rng`: Utilities for key borrowing/splitting used by stochastic measurement flows.
- `photon_weave.core.meta`: Metadata structures that capture dimensions and target ordering to avoid recomputation.

## Meta and Shape Planning
- `DimsMeta` (from `photon_weave.core.meta`) stores static dimensions, target indices, and non-target indices. By reusing an instance, call sites keep shapes stable and skip repeated index lookups—important for JIT and vmap.
- `ShapePlan` (from `photon_weave.state.utils.shape_planning`) pairs `DimsMeta` with the source/target ordering of state objects. Compiled kernels use the plan to reorder once and reuse metadata across repeated operations.
- Typical flow: build a `ShapePlan` (or `DimsMeta`) once, then call the `*_jit_meta` adapters or `jitted` helpers. Adapters move targets forward, kernels operate in that ordered frame, and adapters reshape back—optionally contracting along the way when configured.

## Suggested Next Steps
- Re-enable excluded example pages in the Sphinx build once their formatting is cleaned.
- Extend autosummary coverage to the new core and state utility modules to surface these APIs in the HTML docs.
- Keep metadata-aware call sites using `ShapePlan`/`DimsMeta` for any new operations or measurements to preserve static shapes under JIT.
