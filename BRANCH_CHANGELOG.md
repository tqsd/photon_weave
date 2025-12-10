# Branch Change Log – feature_enhancement vs master

## Scope & Status
- Branch introduces a JIT-oriented core layer (`photon_weave/core/*`), shape-planning helpers, expanded measurement/PNR support, and broader tests/docs. Recent work adds cached contraction kernels and stricter key/threading semantics.
- Status: ready for continued integration; remaining tasks include making ShapePlan the default under JIT and removing Config-based RNG fallbacks on imperative paths.

## New Additions
- **Core layer**: ordered kernels/adapters (`core/adapters.py`, `core/kernels.py`, `core/meta.py`, `core/linear.py`) and cached einsum patterns for apply/measure/trace/Kraus.
- **Shape planning**: `ShapePlan`/`DimsMeta` builders plus `compiled_kernels` helpers to keep shapes static under JIT.
- **Contraction caching**: jitted, cached contraction paths for vector/matrix apply and trace-out, keyed by `(dims, targets, contraction flag)`.
- **Measurement/PNR**: rewritten measurement stack with expectation paths, PNR noise models (efficiency/dark/jitter), and explicit key plumbing; PNR now returns `(outcomes, post, jitter, next_key)`.
- **Config toggle**: `Config.use_jit` enables plan-aware apply/measure paths; guardrails added against dynamic dimension resizing when JIT is enabled.
- **Docs**: architecture/review and JIT/vmap plans, change summaries, and update notes covering the new core/meta plumbing.
- **Tests**: new suites for core adapters/jitted paths, measurement RNG determinism, shape planning, and contraction caching parity.

## Breaking / Behavior Changes
- **Key handling**: jitted measurement/PNR paths now require an explicit PRNG key; several APIs return `next_key` for deterministic chaining. Future work should remove remaining `Config.random_key` fallbacks on non-jitted paths.
- **API returns**: PNR helpers return an extra `next_key`; envelope/composite `measure` can return `(outcomes, next_key)` when `return_key=True`. Call sites expecting only outcomes must adapt if they opt in.
- **Dynamic dimensions under JIT**: Fock apply-operation now raises when `use_jit=True` and `dynamic_dimensions` would resize shapes; callers must precompute/pad cutoffs.
- **Stricter validation**: adapter paths check product-state sizes against dims and operator shapes; invalid shapes now raise `ValueError`/`AssertionError` earlier.

## Highlights to Communicate
- Use `ShapePlan`/`DimsMeta` with `Config.use_jit=True` to keep shapes static and hit cached kernels; contraction caching reduces reorder/reshape overhead for hot paths.
- Measurement/PNR paths are now key-driven and return chaining keys—thread keys explicitly for reproducibility.
- Contraction/invariants: `BaseState.contract` is now abstract (raises) to force concrete implementations; contraction caching covers apply/trace-out, with tests verifying parity vs ordered paths.
- Dynamic resizing is incompatible with JIT; choose fixed cutoffs per compiled run.

## Coverage Notes
- Added tests for adapter/jitted parity, measurement RNG determinism, shape planning, contraction caching, and PNR key handling. Composite measurement/reorder has regression coverage. No regressions expected for legacy imperative paths, but keyless measurement under JIT now errors by design.
