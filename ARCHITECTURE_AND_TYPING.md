# Photon Weave Architecture & Typing Status

## High-Level Architecture
- **Config & RNG**: `photon_weave/photon_weave.py` exposes `Config` (JIT, contraction, seeds). RNG keys come from `core/rng.py`; `__init__.py` pins `JAX_ENABLE_LEGACY_RNG`. Tests run with `JAX_PLATFORMS=cpu`.
- **States**: `state/base_state.py` defines the abstract base; concrete states live in `state/fock.py`, `state/polarization.py`, `state/custom_state.py`. Each holds dimensions, index within a product, expansion level (Label/Vector/Matrix), and data (`state` array or label).
- **Containers**:
  - `state/envelope.py` pairs `Fock` + `Polarization`, manages combination/reordering, apply/measure/trace, and bridges to ops and kernels.
  - `state/composite_envelope.py` combines multiple envelopes into product states.
- **Operations**: `operation/*_operation.py` define enums (`FockOperationType`, `PolarizationOperationType`, `CustomStateOperationType`) plus `operation/operation.py` wrappers. They compute operators, call into `state/utils/operations.py`, which routes to `core/adapters.py`/`core/kernels.py`.
- **Kernels & JIT**: `core/kernels.py` contains pure JAX kernels for apply/measure/trace; `core/jitted.py` wraps them. `state/utils/shape_planning.py` builds `ShapePlan` (dims, target indices) to feed the jitted paths.
- **Measurements**: `state/utils/measurements.py` provides POVM and projective measurement helpers for vectors/matrices and feeds into state methods.
- **Examples & Benchmarks**: `examples/*.py` (e.g., `mach_zehnder_interferometer.py`, `super_dense_coding.py`, `time_bin_encoding.py`) and `benchmarks/*` exercise the public surface—useful for validating behavior when refactoring.

## Current mypy Failure Themes (snapshot)
The code compiles and tests pass, but mypy reports many errors due to interface drift. Main categories:
- **Protocol vs. concrete mismatch**: `BaseStateLike`/`EnvelopeLike` signatures differ from `BaseState`/`Envelope` (measure/apply_kraus args/returns; property types like `expansion_level`, `_num_quanta`, `uid`).
- **Return types from kernels/adapters**: Measurement helpers now return JAX scalars, but adapters are still annotated to return `int` (e.g., `core/adapters.py` and `core/kernels.py` annotations).
- **Enums annotated**: Enum members in `operation/polarization_operation.py` and `operation/fock_operation.py` were annotated; mypy requires unannotated members.
- **Shape planning shims**: Local helpers in `state/utils/shape_planning.py` lack return annotations and accept `object`, causing arg-type errors when calling apply/measure kernels.
- **Measurements utilities**: Inconsistent tuple shapes (Optionals in returns), list mutation typing (`list[Array]` vs. `Sequence`), and missing annotations lead to errors in `state/utils/measurements.py`.
- **Composite/Envelope typing**: Caches (`_plan_cache`) and plan builders expect `BaseState` but receive `BaseStateLike`; `trace_out`/`measure_POVM` arg/return types disagree.
- **BaseState collections**: Lists/dicts typed as concrete `BaseState` conflict with protocol expectations (`BaseStateLike`) across envelope/composite paths.

## Refactor Plan (to reduce circular imports & simplify operations)
Phases are ordered to de-risk changes; run `JAX_PLATFORMS=cpu poetry run pytest` after each chunk and re-run a narrow `mypy` target to track progress.

1. **Stabilize Protocols**
   - Create minimal `interfaces.py` protocols per layer: `BaseStateProto` (dimensions/index/state, `_num_quanta`, `expand`, `measure`, `apply_kraus`, `trace_out`, `resize`, `uid`, `measured`), `EnvelopeProto` (fock, polarization, measure/trace/apply_kraus), `CompositeEnvelopeProto`.
   - Keep protocol methods permissive (`*args`, `**kwargs`) but set consistent return aliases (e.g., `OutcomeMap = dict[BaseStateProto, int]`).
   - Ensure `expansion_level`/`uid` are typed to accept current concrete types (e.g., `ExpansionLevel | None`, `object`).

2. **Align Concrete Classes**
   - Update `BaseState.measure`/`apply_kraus` signatures to match the protocol (allow `*states: BaseStateLike` and return `OutcomeMap`).
   - Make `_num_quanta` a read-only property with concrete implementations in `Fock`/`Polarization`/`CustomState`.
   - Normalize `uid` type to `object` (or `UUID | str`) in concrete classes and in `OutcomeMap` keys.

3. **Re-layer Imports**
   - Enforce dependency direction: `interfaces` → `core` (kernels/adapters/meta) → `state` → `operation` → `envelope/composite` → `examples/tests`.
   - Move any interface-only needs out of `state/utils` so they import protocols, not concretes. For example, `shape_planning` should work on `Sequence[BaseStateProto]` and return `ShapePlan` with only dims/indices, not concrete types.
   - Where circular imports remain, introduce lightweight “view” modules (e.g., `state/typing.py`) that export aliases without importing heavy implementations.

4. **Harmonize Measurement API**
   - Define a single `MeasurementResult = tuple[OutcomeMap, jnp.ndarray | None]` (or similar) and use it across `measure_vector`, `measure_matrix`, `measure_POVM`.
   - In `measurements.py`, remove Optional tails in return tuples where callers expect concrete arrays; cast JAX scalar outcomes to `int` only at API boundaries that require Python ints (tests already handle JAX scalars).
   - Ensure RNG keys are propagated and returned uniformly.

5. **Adapters/Kernels Typing Cleanup**
   - Update return annotations in `core/kernels.py` and `core/adapters.py` to use `jnp.ndarray` (scalar) for outcomes, not `int`.
   - Add explicit return types to small helpers (e.g., local lambdas in `shape_planning`) to quiet mypy without altering behavior.

6. **Envelope/Composite Simplification**
   - Type caches (`_plan_cache`) against `ShapePlan` keys using `Tuple[Tuple[int, int], Tuple[int, int]]` or a dedicated key dataclass to avoid tuple-of-tuple indexing errors.
   - Convert collections in envelope/composite paths to `Sequence[BaseStateLike]` and cast only at boundaries that require concrete states (e.g., resizing).
   - Reduce double-routing: ensure `apply_operation` uses a single path to compute meta and call `apply_operation_vector/matrix` with already-ordered state lists.

7. **Operation Enums & Factory**
   - Keep enum members unannotated; expose typed factories that return `Operation` with fully-typed `operator` fields.
   - Co-locate parameter validation with operator construction to reduce cross-module imports.

8. **Verification Using Examples**
   - Use existing examples as smoke tests for surface APIs:
     - `examples/mach_zehnder_interferometer.py` and `examples/time_bin_encoding.py` cover envelope combine/measure flows.
     - `examples/super_dense_coding.py` exercises composite envelopes and multiple operations.
     - Benchmarks (`benchmarks/lossy_circuit/*.py`, `benchmarks/contraction_vs_jitted.py`) ensure JIT/contraction paths stay performant.
   - After each phase, run a small set of examples with `JAX_PLATFORMS=cpu` to confirm behavior matches current outputs (add assertions in a temporary harness if needed; remove before commit).

9. **Migration Steps**
   - Phase 1: Fix protocols and enum annotations; adjust kernel/adapter return types; rerun `mypy core/`.
   - Phase 2: Align `BaseState`/`Envelope` signatures and measurement utilities; rerun `mypy state/utils/measurements.py state/envelope.py`.
   - Phase 3: Clean up `composite_envelope`/`shape_planning` caches and casts; rerun `mypy state/composite_envelope.py state/utils/shape_planning.py`.
   - Final: Full `mypy` across `photon_weave/` and `pytest` on CPU JAX.

## Quick Insights for Debugging Current Execution
- **RNG determinism**: Seeds are set via `Config.set_seed`; JAX legacy RNG is enabled. Measurement outcomes in tests accept JAX scalar results.
- **JIT vs. dynamic dims**: `Config.use_jit` with `dynamic_dimensions` raises in `Envelope.apply_operation`; resizing is only allowed when JIT is off.
- **Platform**: GitHub Actions and tox set `JAX_PLATFORMS=cpu`; avoid GPU-specific behavior in refactors.

Use this document as the living map for the typing cleanup and architectural hardening. Update sections as refactors land and mypy noise drops.

## Proposed Extensions & API Simplification
The following proposals aim to make the system more modular, interoperable, and pleasant to use, while keeping compatibility for two development rounds before fully adopting the new scheme.

1. **Intermediate Representation (IR) for Interop**
   - Introduce a small, framework-agnostic IR in `core/ir.py`: gates/ops as typed dataclasses (name, params, targets), state specs (dims, basis), and circuits as ordered lists. Keep it JAX-friendly (pytree-compatible) so it can be jitted/vmap’d.
   - Add import/export adapters:
     - **Strawberry Fields**: map Gaussian ops and Fock-level ops to IR; export IR back to SF circuits.
     - **Piquasso**: translate photonic ops and states via IR.
     - **Qiskit**: map polarization/Fock subsets to qubit gates where possible; use IR as the bridge.
   - Keep IR pure-data; no side effects. Concrete simulators (our kernels or external backends) consume IR.

2. **Unique Base Class per Module**
   - Define a single base class per major module folder:
     - `state/base_state.py` → `BaseState` (already present, align to protocol).
     - `operation/base_operation.py` → shared operation base, encapsulating operator caching and parameter validation.
     - `core/base_kernel.py` (optional) → typed protocol for kernels/adapters to reduce import cycles.
   - Within each folder, type against the local base/protocol to avoid cross-imports; export light `interfaces` that re-export these bases to other layers.

3. **Math Decoupling into Core**
   - Move or wrap mathematical primitives into `core/math/`:
     - Operator builders (creation/annihilation, Pauli, rotations, displacements).
     - Contraction utilities and einsum patterns.
     - Measurement probability helpers.
   - State and operation classes call these helpers; avoid inline math in higher layers. This reduces duplication and keeps JAX/pytree constraints localized.

4. **JIT/vmap/Differentiability First**
   - Enforce that new paths are side-effect free and functional: inputs → outputs without in-place Python mutations. Return updated states instead of mutating when in JIT mode; keep legacy mutating paths for compatibility.
   - Ensure all new math lives in JAX-compatible functions; use `jaxtyping`-style annotations (or doc hints) to flag differentiable paths. Provide a small test matrix to validate `jit`, `vmap`, and `grad` on representative ops.

5. **New API Layer with Compatibility Shim**
   - Add a thin fluent API in `api/` (e.g., `api/envelope.py`, `api/circuit.py`) that:
     - Builds envelopes/composites declaratively: `env = EnvelopeBuilder().fock(dim=2, state=1).polarization("R").combine()`.
     - Applies ops via method chaining: `env.apply(FockOp.displace(alpha)).measure(key=...)`.
     - Emits IR under the hood and executes via core kernels.
   - Provide adapters that map the new API calls to legacy `Envelope`/`CompositeEnvelope` methods for two development cycles. Mark legacy interfaces as “compat mode” and plan a removal milestone.

6. **Roadmap Checkpoints**
   - **Phase A (compat)**: Introduce IR, core math wrappers, and new API facades that delegate to existing classes. Keep mutating behavior but add pure variants for JIT/vmap.
   - **Phase B (migration)**: Flip default paths to pure/IR-driven execution; keep legacy shims emitting deprecation warnings.
   - **Phase C (cleanup)**: Remove legacy shims; enforce protocol alignment and mypy cleanliness; examples updated to new API.

## Refactoring Direction (Envelope-First Simplicity)
- Preserve the public `Envelope` and `CompositeEnvelope` interfaces exactly as described in the paper (combine, apply, measure, trace_out, metadata such as wavelength/temporal profile). They remain the primary user abstraction and signature surface.
- Behind those interfaces, map calls onto the lightweight circuit/IR path (`CircuitSpec` + planner + kernels). The envelopes become facades that assemble IR from their state containers and forward to the runtime executor, keeping behavior while reducing internal coupling.
- Keep typing minimal and concrete: prefer small data records (`StateSpec`, `OpSpec`) and runtime assertions over Protocol-heavy checks; avoid `Any`/`object` except at external boundaries. This keeps serialization easy and JAX-friendly.
- Shape planning should consume only dimension/target tuples, not concrete envelope classes, so the IR layer stays pure-data and easy to serialize/compile.
- Compatibility: during Phase A/B, continue to route the existing envelope/composite methods to the new runtime while tests/examples validate parity; deprecation only applies to internal helpers, not the envelope APIs.

## Naming & Circuit Mapping
To keep a single, memorable user-facing name while mapping cleanly to the circuit-oriented internals:
- Expose the public API under the existing package name (`photon_weave`) but brand the circuit DSL as **Weave Circuits**. Users import from `photon_weave.weave` (e.g., `WeaveCircuit`, `WeaveOp`, `WeaveState`).
- Internally, `WeaveCircuit` is a light facade over the IR (`CircuitSpec`, `OpSpec`, `StateSpec`); constructors only assemble these specs and never own kernels.
- Execution routes through the runtime executor (`execute_circuit`), which translates specs to planner/kernels. This preserves the unique naming for users while keeping internals pure-data and circuit-behavior driven.
- Interop layers (`interop/*`) accept or emit Weave Circuits, so external frameworks only learn one name while mapping to IR remains an implementation detail.

These changes, combined with the earlier typing refactor plan, should yield a cleaner, interoperable, and JAX-friendly architecture with a gentle migration path for users and downstream frameworks.

## Execution Profiling Snapshot (Mach–Zehnder Example)
Timing recorded on CPU with `JAX_PLATFORMS=cpu` for the core steps inside `mach_zender_probabilities`:

- Cold start (phase=0.0): env build 0.06 ms; op build 0.015 ms; composite build 0.033 ms; first beam splitter apply ~718 ms (JAX/XLA compile cold-start + contraction setup); phase shift apply ~68 ms; second beam splitter apply ~2.7 ms; trace-outs ~0.18 ms.
- Warm cache (subsequent phases): env/op/composite each <0.05 ms; first beam splitter apply ~3–4 ms; phase shift ~2 ms; second beam splitter ~2–3 ms; trace-outs ~0.17 ms.

Findings:
- The dominant cost is the first `apply_operation` (JAX compilation and contraction path). Subsequent calls reuse compiled kernels and drop to low single-digit milliseconds.
- Python-layer overhead (Operation → ShapePlan → adapters → kernels) is negligible after warm-up, but batching ops into a single IR execution could reduce Python hops further.
- Practical guidance: perform a warm-up call before benchmarking; cache ShapePlans/compiled kernels aggressively; consider a pure functional “apply IR sequence” path to amortize setup.

## Progress Checkpoint (Runtime & Envelope Integration)
- IR + runtime are live: `StateSpec`/`OpSpec`/`MeasureSpec`/`TraceOutSpec` plus `execute_circuit` route through planner/kernels without touching envelope classes.
- Envelope and CompositeEnvelope now emit IR internally for `apply_operation` while preserving their public API and paper-described semantics.
- Planner now supports dims-only planning (`plan_from_dims`) so core layers can operate without importing state objects.
- Next up: extend IR delegation to envelope/composite measurement and trace-out flows to remove residual state↔operation coupling.

## Concrete Delivery Plan (preserve Envelope interface)
- **Scaffold IR/runtime (Phase A)**: add `core/ir.py` (`StateSpec`, `OpSpec`, `CircuitSpec`) and `runtime/execute_circuit` that call existing kernels via the planner. Keep IR pure-data and JSON-serializable.
- **Planner extraction**: move `shape_planning` helpers to accept only dims/targets; cache plans by `(dims, targets)`. Ensure JIT paths are shape-stable.
- **Envelope/Composite facades**: keep current public methods/metadata; internally translate to IR and dispatch to runtime. Maintain combine/apply/measure/trace_out signatures and behaviors from the paper.
- **Operation builders**: refactor `*_operation.py` to produce `OpSpec` + operator matrices; enums stay unannotated. Add small validation helpers; remove protocol dependencies.
- **Measurements/RNG unification**: normalize measurement returns across vector/matrix/POVM; pass/return RNG keys explicitly; avoid Optional tails and widen scalar handling to JAX arrays.
- **Interop adapters**: add `interop/` shims that map Weave Circuits (IR) to StrawberryFields/Piquasso/Qiskit and back; keep Envelope-facing API untouched.
- **Testing & parity**: wire examples/benchmarks through the new runtime via envelopes; add snapshot tests for IR JSON round-trip and a few deterministic measurement cases on CPU. Use examples as smoke tests after each milestone.
