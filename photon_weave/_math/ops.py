import numba as nb
import numpy as np
from numba import njit
import jax.numpy as jnp
from jax import jit
import jax
from typing import Union, List


@njit('complex128[:,::1](uintc)', cache=True, parallel=True, fastmath=True)
def annihilation_operator(cutoff: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, cutoff, dtype=np.complex128)), 1)


@njit('complex128[::1,:](uintc)', cache=True, parallel=True, fastmath=True)
def creation_operator(cutoff: int)-> np.ndarray:
    return np.conjugate(annihilation_operator(cutoff=cutoff)).T


def matrix_power(mat: np.ndarray, power: int) -> np.ndarray:
    if power == 1:
        return mat
    elif power == 2:
        return np.dot(mat, mat)
    elif power == 3:
        return mat @ mat @ mat
    else:
        return np.linalg.matrix_power(mat, power)


@njit('complex128[:,::1](complex128[:,::1])', cache=True, parallel=True, fastmath=True)
def _expm(mat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(mat)
    return eigvecs @ np.diag(np.exp(eigvals)) @ np.linalg.pinv(eigvecs)


@njit('complex128[:,::1](complex128, uintc)', cache=True, parallel=True, fastmath=True)
def squeezing_operator(zeta: complex, cutoff: int):
    create = creation_operator(cutoff=cutoff)
    destroy = annihilation_operator(cutoff=cutoff)
    operator = 0.5 * (
        np.conj(zeta) * (destroy @ destroy) - zeta * (create @ create)
    )
    return _expm(operator)


@njit('complex128[:,::1](complex128, uintc)', cache=True, parallel=True, fastmath=True)
def displacement_operator(alpha: complex, cutoff: int):
    create = creation_operator(cutoff=cutoff)
    destroy = annihilation_operator(cutoff=cutoff)
    operator = alpha * create - alpha * destroy
    return _expm(operator)


@njit(cache=True, parallel=True, fastmath=True)
def phase_operator(theta: float, cutoff: int):
    return np.diag([np.exp(1j * n * theta) for n in range(cutoff)])


# to do: implement beamsplitter here
@jit
def compute_einsum(einsum_str: str,
                   *operands: Union[jax.Array, np.ndarray]) -> jax.Array:
    """
    Computes einsum using the provided einsum_str and matrices
    with the gpu if accessible (jax.numpy).
    Parameters
    ----------
    einsum_str: str
        Einstein Sum String
    operatnds: Union[jax.Array, np.ndarray]
        Operands for the einstein sum
    Returns
    -------
    jax.Array
        resulting matrix after eintein sum
    """
    return jnp.einsum(einsum_str, *operands)

@jit
def apply_kraus(
        density_matrix: Union[np.ndarray, jnp.array],
        kraus_operators: List[Union[np.ndarray, jnp.array]]) -> jnp.array:
    """
    Apply Kraus operators to the density matrix.
    Parameters
    ----------
    density_matrix: Union[np.ndarray, jnp.array]
        Density matrix onto which the Kraus operators are applied
    kraus_operators: List[Union[np.ndarray, jnp.array]]
        List of Kraus operators to apply to the density matrix
    """
    new_density_matrix = jnp.zeros_like(density_matrix)
    for K in kraus_operators:
        new_density_matrix += K @ density_matrix @ jnp.conjugate(K).T

    return new_density_matrix

@jit
def kraus_identity_check(operators: List[Union[np.ndarray, jnp.array]], tol: float = 1e-6) -> bool:
    """
    Check if Kraus operators sum to the identity matrix.

    Parameters
    ----------
    kraus_operators: List[np.ndarray, jnp.array]
        List of the operators
    tol: float
        Tolerance for the floating-point comparisons

    Returns
    -------
    bool
        True if the Kraus operators sum to identity within the tolerance
    """
    dim = operators[0].shape[0]
    identity_matrix = jnp.eye(dim)
    sum_kraus = sum(
        jnp.matmul(jnp.conjugate(K.T),K) for K in operators
    )
    return jnp.allclose(sum_kraus, identity_matrix, atol=tol)
