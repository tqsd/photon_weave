"""
Benchmarks: legacy path vs JIT/plan-aware path.

Scenarios:
1) Single-envelope feature circuit (adapted from examples/quantum_feature_classifier.py)
   using displacement + phase shift and photon-number expectation.
2) Composite with two envelopes, beam splitter + phase shift, then joint measurement.

For each scenario we run both the legacy (use_jit=False) and JIT (use_jit=True)
paths, average 5 runs, and plot the timings and speedups.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Ensure we import the in-repo version
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photon_weave.operation import (
    CompositeOperationType,
    FockOperationType,
    Operation,
)
from photon_weave.photon_weave import Config
from photon_weave.state.envelope import Envelope
from photon_weave.state.utils.measurements import (
    measure_matrix,
    measure_matrix_jit,
)
from photon_weave.state.utils.operations import apply_operation_matrix
from photon_weave.state.utils.shape_planning import build_plan

RUNS = 20
SAVE_DIR = Path("benchmarks")


@dataclass(frozen=True)
class StateStub:
    dimensions: int


def _basis_density(dims: Tuple[int, ...], index: int = 0) -> jnp.ndarray:
    size = int(jnp.prod(jnp.array(dims)))
    vec = jnp.zeros((size, 1), dtype=jnp.complex128)
    vec = vec.at[index, 0].set(1.0)
    return vec @ jnp.conj(vec.T)


def _time_runs(fn: Callable[[], object], runs: int = RUNS) -> float:
    """Run fn `runs` times and return average duration in seconds."""
    # Warm-up
    fn()
    start = time.perf_counter()
    for _ in range(runs):
        out = fn()
        # Block if JAX array
        if isinstance(out, jnp.ndarray):
            jax.block_until_ready(out)
    duration = time.perf_counter() - start
    return duration / runs


def _feature_ops_sequence(feature: float) -> Tuple[Operation, ...]:
    """Heavier sequence of operations to stress apply paths."""
    return (
        Operation(FockOperationType.Displace, alpha=0.5 + 0.3 * feature),
        Operation(FockOperationType.PhaseShift, phi=0.1 + 0.5 * feature),
        Operation(FockOperationType.Displace, alpha=-0.2 + 0.4 * feature),
        Operation(FockOperationType.PhaseShift, phi=-0.3 + 0.2 * feature),
        Operation(FockOperationType.Squeeze, zeta=0.2 * (1 + 1j)),
        Operation(FockOperationType.Displace, alpha=0.1 - 0.25 * feature),
    )


def single_envelope_once(feature: float, dims: int) -> jnp.ndarray:
    """Apply a sequence of operations then compute photon-number expectation."""
    env = Envelope()
    env.fock.dimensions = dims
    env.fock.state = 0
    for op in _feature_ops_sequence(feature):
        env.apply_operation(op, env.fock)
    probs, _ = env.fock.measure_expectation()
    levels = jnp.arange(probs.shape[0], dtype=probs.dtype)
    return jnp.dot(levels, probs)


def run_single_envelope(use_jit: bool) -> float:
    cfg = Config()
    cfg._use_jit = use_jit  # internal toggle for this branch
    cfg.set_contraction(True)
    dims = 12
    features = jnp.linspace(-1.0, 1.0, 8)

    def fn() -> jnp.ndarray:
        def body(fval):
            return single_envelope_once(fval, dims)

        return jax.vmap(body)(features)

    if use_jit:
        compiled = jax.jit(fn)
        compiled()  # compile
        jax.block_until_ready(compiled())
        return _time_runs(compiled)
    return _time_runs(fn)


def composite_plan_benchmark(use_jit: bool) -> float:
    """
    Benchmark composite-style operations and measurements using matrix evolution
    with and without ShapePlan.
    """
    cfg = Config()
    cfg._use_jit = use_jit
    cfg.set_contraction(True)
    dims = (10, 10)
    states = [StateStub(d) for d in dims]
    targets = tuple(states)
    plan_ops = build_plan(states, targets)
    plan_meas = build_plan(states, [states[0]])

    # Initial density matrix |1,0>
    rho0 = _basis_density(dims, index=1)

    bs = Operation(
        CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4
    )
    bs.dimensions = list(dims)
    bs_op = bs.operator

    reps = 12

    def legacy_apply():
        rho = rho0
        for _ in range(reps):
            rho = apply_operation_matrix(
                states, targets, rho, bs_op, use_contraction=True
            )
        return rho

    def jitted_apply():
        rho = rho0
        for _ in range(reps):
            rho = apply_operation_matrix(
                states,
                targets,
                rho,
                bs_op,
                meta=plan_ops,
                use_contraction=True,
            )
        return rho

    apply_legacy_avg = _time_runs(legacy_apply)

    apply_jit_fn = jax.jit(jitted_apply) if use_jit else jitted_apply
    # warm and time apply
    apply_jit_fn()
    if use_jit:
        jax.block_until_ready(apply_jit_fn())
    apply_jit_avg = _time_runs(apply_jit_fn)

    # Use a fixed rho snapshot for measurement timing to isolate measurement cost.
    rho_for_meas = jitted_apply()

    def legacy_meas():
        key = jax.random.PRNGKey(0)
        outcomes, post = measure_matrix(
            states, [states[0]], rho_for_meas, key=key
        )
        return post

    def jitted_meas():
        key = jax.random.PRNGKey(0)
        key = jax.random.PRNGKey(0)
        outcomes, post, _ = measure_matrix_jit(
            states, [states[0]], rho_for_meas, key=key, meta=plan_meas
        )
        return post

    meas_legacy_avg = _time_runs(legacy_meas)
    meas_jitted_avg = _time_runs(jitted_meas if use_jit else legacy_meas)

    return apply_legacy_avg, apply_jit_avg, meas_legacy_avg, meas_jitted_avg


@dataclass
class BenchmarkResult:
    label: str
    legacy_avg: float
    jit_avg: float
    kind: str = "apply"

    @property
    def speedup(self) -> float:
        return self.legacy_avg / self.jit_avg if self.jit_avg > 0 else 0.0


def plot_results(results: Tuple[BenchmarkResult, ...]) -> None:
    labels = [r.label for r in results]
    legacy = [r.legacy_avg for r in results]
    jitted = [r.jit_avg for r in results]
    speedups = [r.speedup for r in results]

    x = jnp.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    ax0.bar(x - width / 2, legacy, width, label="legacy")
    ax0.bar(x + width / 2, jitted, width, label="jitted")
    ax0.set_ylabel("Avg runtime (s)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=10)
    ax0.legend()
    ax0.set_title(f"Average of {RUNS} runs")

    ax1 = axes[1]
    ax1.bar(labels, speedups, color="#4caf50")
    ax1.set_ylabel("Speedup (legacy / jitted)")
    ax1.set_title("Speedup")
    for idx, val in enumerate(speedups):
        ax1.text(idx, val + 0.02, f"{val:.2f}x", ha="center", va="bottom")

    fig.tight_layout()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / "contraction_vs_jitted.png"
    plt.savefig(out_path, dpi=180)
    print(f"Saved plot to {out_path}")


def main() -> None:
    single_legacy = run_single_envelope(use_jit=False)
    single_jit = run_single_envelope(use_jit=True)

    comp_apply_legacy, comp_apply_jit, comp_meas_legacy, comp_meas_jit = (
        composite_plan_benchmark(use_jit=True)
        if True
        else composite_plan_benchmark(use_jit=False)
    )

    results = (
        BenchmarkResult("single envelope (apply)", single_legacy, single_jit),
        BenchmarkResult("composite apply", comp_apply_legacy, comp_apply_jit),
        BenchmarkResult(
            "composite measure",
            comp_meas_legacy,
            comp_meas_jit,
            kind="measure",
        ),
    )

    for r in results:
        print(
            f"{r.label:>20}: legacy {r.legacy_avg:.4f}s, "
            f"jitted {r.jit_avg:.4f}s, speedup {r.speedup:.2f}x"
        )

    plot_results(results)


if __name__ == "__main__":
    main()
