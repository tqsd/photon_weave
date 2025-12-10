import jax
import jax.numpy as jnp

from photon_weave.core import adapters, linear


def _basis_vector(index: int, dim: int) -> jnp.ndarray:
    vec = jnp.zeros((dim, 1), dtype=jnp.complex128)
    vec = vec.at[index, 0].set(1)
    return vec


def _density(vec: jnp.ndarray) -> jnp.ndarray:
    return vec @ jnp.conj(vec.T)


def test_apply_operation_vector_contraction_matches_ordered():
    dims = (2, 2)
    target_indices = (0,)
    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    psi = _basis_vector(1, 4)  # |01>

    contracted = adapters.apply_operation_vector(
        dims, target_indices, psi, x, use_contraction=True
    )
    ordered = adapters.apply_operation_vector(
        dims, target_indices, psi, x, use_contraction=False
    )

    assert jnp.allclose(contracted, ordered)


def test_apply_operation_matrix_contraction_matches_ordered():
    dims = (2, 2)
    target_indices = (0,)
    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    psi = _basis_vector(1, 4)
    rho = _density(psi)

    contracted = adapters.apply_operation_matrix(
        dims, target_indices, rho, x, use_contraction=True
    )
    ordered = adapters.apply_operation_matrix(
        dims, target_indices, rho, x, use_contraction=False
    )
    assert jnp.allclose(contracted, ordered)


def test_trace_out_matrix_contraction_matches_ordered():
    dims = (2, 2)
    target_indices = (1,)
    psi = _basis_vector(1, 4)
    rho = _density(psi)

    contracted = adapters.trace_out_matrix(
        dims, target_indices, rho, use_contraction=True
    )
    ordered = adapters.trace_out_matrix(
        dims, target_indices, rho, use_contraction=False
    )
    assert jnp.allclose(contracted, ordered)


def test_measure_vector_deterministic_with_key():
    dims = (2,)
    target_indices = (0,)
    psi = _basis_vector(1, 2)
    key = jax.random.PRNGKey(0)
    outcome, post, _ = linear.measure_vector(dims, target_indices, psi, key)
    assert outcome == 1
    assert post.shape == (1, 1)


def test_measure_matrix_deterministic_with_key():
    dims = (2,)
    target_indices = (0,)
    rho = _density(_basis_vector(0, 2))
    key = jax.random.PRNGKey(1)
    outcome, post, _ = linear.measure_matrix(dims, target_indices, rho, key)
    assert outcome == 0
    assert post.shape == (1, 1)
