# Branch Change Log — feature_enhancement vs master

This page summarizes the deltas between `feature_enhancement` and `master`, highlights breaking changes, and notes the major additions to the codebase and docs.

## What’s New
- **JIT-oriented core layer**: ordered kernels/adapters (`photon_weave/core/*`), static `DimsMeta` helpers, and cached einsum patterns for apply/measure/trace/Kraus.
- **Shape planning**: `ShapePlan` builders plus `compiled_kernels` wrappers to keep shapes static under JIT.
- **Contraction caching**: jitted, cached contraction paths for vector/matrix apply and trace-out keyed by `(dims, targets, contraction flag)`.
- **Measurement/PNR updates**: expectation paths, PNR noise models (efficiency/dark/jitter), and explicit key plumbing. PNR now returns `(outcomes, post, jitter, next_key)`.
- **Config/use_jit toggle**: enables plan-aware apply/measure paths; dynamic resizing is blocked under JIT.
- **Tests/docs**: new suites for adapters/jitted paths, measurement RNG determinism, shape planning, contraction caching, and updated architecture/JIT plan docs.

## Breaking/Behavior Changes
- **Key requirements**: jitted measurement/PNR paths require an explicit PRNG key; some APIs can now return `next_key` for deterministic chaining.
- **API returns**: PNR helpers add a `next_key`; envelope/composite `measure` can opt into returning `(outcomes, next_key)` (`return_key=True`).
- **Dynamic dimensions vs JIT**: applying ops with `use_jit=True` rejects dynamic resizing—choose fixed cutoffs before compiling.
- **Stricter validation**: adapter paths validate operator shapes and product-state sizes up front and raise clear errors on mismatch.

## Notes for Users
- Prefer `ShapePlan`/`DimsMeta` with `Config.use_jit=True` to keep shapes static and hit cached kernels.
- Thread keys explicitly through measurement/PNR; rely on returned `next_key` when chaining calls.
- For contraction-heavy workloads, enable contraction and let the cached kernels handle the einsum path to reduce reorder/reshape overhead.
