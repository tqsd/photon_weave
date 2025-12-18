# Changelog

All notable changes to this project will be documented in this file. Releases are managed automatically via semantic-release using conventional commit messages.

## [0.2.0] - 2025-12-18

### Added
- JIT-oriented core layer with ordered adapters/kernels, `ShapePlan` helpers, and cached contraction paths for apply/measure/trace flows.
- Expanded measurement/PNR utilities with expectation paths, noise models (efficiency/dark/jitter), and explicit PRNG key threading.
- Documentation navigation and architecture notes are now part of the Sphinx build (MyST-enabled, architecture/typing map included).

### Changed
- `Config.use_jit=True` now rejects dynamic dimension resizing; callers must supply fixed cutoffs.
- Measurement and PNR helpers can return `(outcomes, next_key)` when key chaining is enabled, aligning with the jitted execution model.
- Sphinx docs now read the package version directly from `pyproject.toml` to avoid drift between releases.
- Core apply paths cache opt_einsum expressions and reuse ShapePlan-aware JIT even when contraction is enabled, reducing retracing and Python overhead.
- Measurement helpers accept precomputed target indices and fix indexing for combined vector/matrix cases, restoring correct probabilities with provided keys.

### Breaking
- Jitted measurement/PNR paths require explicit PRNG keys; legacy implicit key usage is no longer accepted when JIT is enabled.
