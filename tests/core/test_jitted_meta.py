import jax
import jax.numpy as jnp

from photon_weave.core import adapters
from photon_weave.core.meta import make_meta


def _basis_state(idx: int, dim: int) -> jnp.ndarray:
    vec = jnp.zeros((dim, 1), dtype=jnp.complex128)
    vec = vec.at[idx, 0].set(1.0)
    return vec


def test_apply_op_vector_jit_meta_matches_adapter():
    dims = (2, 2)
    meta = make_meta(dims, (0,))
    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    state = _basis_state(0, 4)  # |00>
    out_jit = adapters.apply_operation_vector_jit_meta(meta, state, x)
    out_ref = adapters.apply_operation_vector(dims, (0,), state, x)
    assert jnp.allclose(out_jit, out_ref)


def test_measure_vector_jit_meta_deterministic_zero():
    dims = (2, 2)
    meta = make_meta(dims, (0,))
    state = _basis_state(0, 4)  # |00>
    key = jax.random.PRNGKey(0)
    outcome, post, _ = adapters.measure_vector_jit_meta(meta, state, key)
    assert outcome == 0
    assert post.shape == (2, 1)
    assert jnp.allclose(post, jnp.array([[1.0], [0.0]], dtype=post.dtype))


def test_trace_out_matrix_jit_meta():
    dims = (2, 2)
    meta = make_meta(dims, (0,))
    state = _basis_state(0, 4)
    rho = state @ jnp.conj(state).T
    reduced = adapters.trace_out_matrix_jit_meta(meta, rho)
    assert reduced.shape == (2, 2)
    assert jnp.allclose(reduced, jnp.array([[1.0, 0.0], [0.0, 0.0]]))
