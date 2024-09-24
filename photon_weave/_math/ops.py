import numba as nb
import numpy as np
from numba import njit
import jax.numpy as jnp
from jax import jit
import jax
from typing import Union, List
from jax.scipy.linalg import expm
jax.config.update("jax_enable_x64", True)

def annihilation_operator(cutoff: int) -> jnp.ndarray:
    return jnp.diag(jnp.sqrt(jnp.arange(1, cutoff, dtype=np.complex128)), 1)

def creation_operator(cutoff: int)-> jnp.ndarray:
    return jnp.conjugate(annihilation_operator(cutoff=cutoff)).T

def _expm(mat: jnp.ndarray) -> np.ndarray:
    eigvals, eigvecs = jnp.linalg.eig(mat)
    return eigvecs @ jnp.diag(jnp.exp(eigvals)) @ jnp.linalg.pinv(eigvecs)

def squeezing_operator(cutoff: int, zeta:complex) -> jnp.ndarray:
    create = creation_operator(cutoff=cutoff)
    destroy = annihilation_operator(cutoff=cutoff)
    operator = 0.5 * (
        jnp.conj(zeta) * (destroy @ destroy) - zeta * (create @ create)
    )
    return expm(operator)


def displacement_operator(cutoff:int, alpha: complex) -> jnp.ndarray:
    create = creation_operator(cutoff=cutoff)
    destroy = annihilation_operator(cutoff=cutoff)
    operator = alpha * create - jnp.conj(alpha) * destroy
    return expm(operator)


def phase_operator(cutoff:int, theta: float) -> jnp.ndarray:
    """
    Returns a phase shift operator, given the dimensions

    .. math::
      \hat{R}(\theta) = \sum_{n=0}^{\text{cutoff}-1} e^{1 n \theta }|n\rangle \langle n|

    This operator applies a phase shift to each Fock state |n⟩ proportional to the integer n.
    The phase shift is given by :math:`e^{i n \theta}`, where `theta` is the phase shift parameter.
    

    Parameters
    ----------
    cutoff: int
        Cutoff dimensions
    theta: float
        Phase shift for the operator

    Returns
    -------
    jnp.ndarray
        Constructed opreator

    Notes
    -----
    The phase shift operator is unitary and is used to rotate the phase of a quantum state
    in the Fock basis. The diagonal matrix elements are complex exponentials that apply a
    phase proportional to the Fock state number.
    """
    return jnp.diag(jnp.array([jnp.exp(1j * n * theta) for n in range(cutoff)]))


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
        density_matrix: Union[np.ndarray, jnp.ndarray],
        kraus_operators: List[Union[np.ndarray, jnp.ndarray]]) -> jnp.ndarray:
    """
    Apply Kraus operators to the density matrix.
    Parameters
    ----------
    density_matrix: Union[np.ndarray, jnp.ndarray]
        Density matrix onto which the Kraus operators are applied
    kraus_operators: List[Union[np.ndarray, jnp.ndarray]]
        List of Kraus operators to apply to the density matrix
    Returns
    jnp.ndarray
        density matrix after applying Kraus operators
    """
    new_density_matrix = jnp.zeros_like(density_matrix)
    for K in kraus_operators:
        new_density_matrix += K @ density_matrix @ jnp.conjugate(K).T

    return new_density_matrix

@jit
def kraus_identity_check(operators: List[Union[np.ndarray, jnp.ndarray]], tol: float = 1e-6) -> bool:
    """
    Check if Kraus operators sum to the identity matrix.

    Parameters
    ----------
    kraus_operators: List[np.ndarray, jnp.Array]
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

@jit
def normalize_vector(vector: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
    """
    Normalizes the given vector and returns it
    Parameters
    ----------
    vector: Union[jnp.ndarray, np.ndarray]
        Vector which should be normalied
    """
    trace = jnp.trace(vector)
    return vector/trace

@jit
def normalize_matrix(vector: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
    """
    Normalizes the given matrix and returns it
    Parameters
    ----------
    vector: Union[jnp.ndarray, np.ndarray]
        Vector which should be normalied
    """
    norm = jnp.linalg.norm(vector)
    return vector/norm

def num_quanta_vector(vector: Union[jnp.ndarray, np.ndarray]) -> int:
    """
    Returns highest possible measurement outcome
    Parameters
    ----------
    vector: Union[jnp.ndarray, np.ndarray]
        vector for which the max possible quantua has to be calculated
    """
    non_zero_indices = jnp.nonzero(vector)[0]
    return non_zero_indices[-1]

def num_quanta_matrix(matrix: Union[jnp.ndarray, np.ndarray]) -> int:
    """
    Returns highest possible measurement outcome
    Parameters
    ----------
    matrix: Union[jnp.ndarray, np.ndarray]
        matrix for which the max possible quantua has to be calculated
    """
    non_zero_rows = jnp.any(matrix != 0, axis=1)
    non_zero_cols = jnp.any(matrix != 0, axis=0)

    highest_non_zero_index_row = (
        jnp.where(non_zero_rows)[0][-1] if jnp.any(non_zero_rows) else None
    )
    highest_non_zero_index_col = (
        jnp.where(non_zero_cols)[0][-1] if jnp.any(non_zero_cols) else None
    )
    # Determine the overall highest index
    highest_non_zero_index_matrix = max(
        highest_non_zero_index_row, highest_non_zero_index_col
    )
    return highest_non_zero_index_matrix
