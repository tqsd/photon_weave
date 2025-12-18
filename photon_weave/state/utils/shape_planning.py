"""
Helpers to build reusable dimension metadata for JIT-friendly paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from photon_weave.core import jitted
from photon_weave.core.meta import DimsMeta, make_meta
from photon_weave.photon_weave import Config
from photon_weave.state.interfaces import BaseStateLike as BaseState
from photon_weave.state.interfaces import CompositeEnvelopeLike as CompositeEnvelope
from photon_weave.state.interfaces import EnvelopeLike as Envelope
from photon_weave.state.interfaces import (
    OperationLike,
)


@dataclass(frozen=True)
class ShapePlan:
    dims: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    rest_indices: Tuple[int, ...]
    meta: DimsMeta


def build_plan(
    state_objs: Sequence[BaseState], target_states: Sequence[BaseState]
) -> ShapePlan:
    dims = tuple(s.dimensions for s in state_objs)
    target_indices = tuple(state_objs.index(s) for s in target_states)
    meta = make_meta(dims, target_indices)
    return ShapePlan(
        dims=dims,
        target_indices=target_indices,
        rest_indices=meta.rest_indices,
        meta=meta,
    )


def build_meta(
    state_objs: Sequence[BaseState], target_states: Sequence[BaseState]
) -> DimsMeta:
    dims = tuple(s.dimensions for s in state_objs)
    target_indices = tuple(state_objs.index(s) for s in target_states)
    return make_meta(dims, target_indices)


def ensure_static_dims_for_ops(
    state_objs: Sequence[BaseState], operations: Sequence[OperationLike]
) -> None:
    """
    Lightweight helper to pre-size Fock states for a sequence of operations
    before entering JIT/static mode.
    """
    from photon_weave.operation.fock_operation import FockOperationType

    for op in operations:
        for so in state_objs:
            if isinstance(op._operation_type, FockOperationType):
                if getattr(so, "_num_quanta", None) is None:
                    continue
                to = so.trace_out()
                op.compute_dimensions(so._num_quanta, to)
                if so.dimensions != op.dimensions[0]:
                    so.resize(op.dimensions[0])


def envelope_plan(envelope: Envelope, *target_states: BaseState) -> ShapePlan:
    """
    Build a ShapePlan for an envelope (fock + polarization).
    If no targets are provided, all subsystems are targets.
    """
    state_objs = [envelope.fock, envelope.polarization]
    targets = target_states if target_states else state_objs
    return build_plan(state_objs, targets)


def composite_plan(
    composite_envelope: CompositeEnvelope,
    product_state_index: int = 0,
    *target_states: BaseState,
) -> ShapePlan:
    ps = composite_envelope.product_states[product_state_index]
    state_objs = list(ps.state_objects)
    targets = target_states if target_states else state_objs
    return build_plan(state_objs, targets)


def compiled_kernels(plan: ShapePlan):
    """
    Return callables bound to the given plan/meta for jitted execution.
    """

    def apply_op_vec(ps, op, use_contraction=False):
        return jitted.apply_operation_vector(
            plan.meta, ps, op, use_contraction=use_contraction
        )

    def apply_op_mat(ps, op, use_contraction=False):
        return jitted.apply_operation_matrix(
            plan.meta, ps, op, use_contraction=use_contraction
        )

    def apply_kraus(ps, ops):
        return jitted.apply_kraus_matrix(
            plan.meta, ps, ops, use_contraction=Config().contractions
        )

    def measure_vec(ps, key):
        return jitted.measure_vector(plan.meta, ps, key)

    def measure_mat(ps, key):
        return jitted.measure_matrix(plan.meta, ps, key)

    def measure_povm(ops, ps, key):
        return jitted.measure_povm_matrix(plan.meta, ops, ps, key)

    def trace_out(ps, use_contraction=False):
        return jitted.trace_out_matrix(plan.meta, ps, use_contraction=use_contraction)

    class Kernels:
        apply_op_vector = staticmethod(apply_op_vec)
        apply_op_matrix = staticmethod(apply_op_mat)
        apply_kraus_matrix = staticmethod(apply_kraus)
        measure_vector = staticmethod(measure_vec)
        measure_matrix = staticmethod(measure_mat)
        measure_povm_matrix = staticmethod(measure_povm)
        trace_out_matrix = staticmethod(trace_out)

    return Kernels()
