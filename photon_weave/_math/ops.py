import numpy as np
from numba import njit


@njit
def annihilation_operator(cutoff: int) -> np.ndarray[np.complex128]:
    return np.diag(np.sqrt(np.arange(1, cutoff, dtype=np.complex128)), 1)


@njit
def creation_operator(cutoff: int) -> np.ndarray[np.complex128]:
    return np.conjugate(annihilation_operator(cutoff=cutoff)).T


def matrix_power(mat: np.ndarray, power: int) -> np.ndarray:
    return np.linalg.matrix_power(mat, power)


@njit(fastmath=True)
def _expm(mat: np.ndarray):
    eigvals, eigvecs = np.linalg.eig(mat)
    return eigvecs @ np.diag(np.exp(eigvals)) @ np.linalg.pinv(eigvecs)


@njit
def squeezing_operator(zeta: complex, cutoff: int):
    create = creation_operator(cutoff=cutoff)
    destroy = annihilation_operator(cutoff=cutoff)
    operator = 0.5 * (
        np.conj(zeta) * np.dot(destroy, destroy) - zeta * np.dot(create, create)
    )
    return _expm(operator)


@njit
def displacement_operator(alpha: complex, cutoff: int):
    create = creation_operator(cutoff=cutoff)
    destroy = annihilation_operator(cutoff=cutoff)
    operator = alpha * create - alpha * destroy
    return _expm(operator)


@njit
def phase_operator(theta: float, cutoff: int):
    return np.diag([np.exp(1j * n * theta) for n in range(cutoff)])


# to do: implement beamsplitter here    