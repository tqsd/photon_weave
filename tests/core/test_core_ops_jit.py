import jax
import jax.numpy as jnp

from photon_weave.core import ops


def test_x_operator_jittable_matches_explicit():
    x = ops.x_operator()
    x_jit = jax.jit(ops.x_operator)()
    assert jnp.allclose(x, x_jit)


def test_rx_operator_grad_and_jit():
    def trace_real(theta):
        mat = ops.rx_operator(theta)
        return jnp.real(jnp.trace(mat))

    grad_fn = jax.jit(jax.grad(trace_real))
    val = grad_fn(0.1)
    assert jnp.isfinite(val)


def test_fidelity_jit():
    psi = jnp.array([1.0, 0.0], dtype=jnp.complex64)
    phi = jnp.array([0.0, 1.0], dtype=jnp.complex64)
    fid = jax.jit(ops.fidelity)(psi, phi)
    assert jnp.allclose(fid, 0.0)


def test_compute_einsum_matches_contract():
    a = jnp.arange(4).reshape(2, 2)
    b = jnp.arange(4).reshape(2, 2)
    out = ops.compute_einsum("ij,jk->ik", a, b)
    expected = jnp.array([[2, 3], [6, 11]])
    assert jnp.allclose(out, expected)
