import numba as nb
import numpy as np
from numba import njit
import jax.numpy as jnp
from jax import jit
from typing import Union

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
