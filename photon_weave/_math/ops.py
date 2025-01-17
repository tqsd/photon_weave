from typing import List, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
jitted_exp = jit(expm)


@jit
def identity_operator() -> jax.Array:
    """
    identity_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.eye(N=2)


@jit
def hadamard_operator() -> jax.Array:
    """
    hadamard_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[1, 1], [1, -1]]) * (1 / jnp.sqrt(2))


@jit
def x_operator() -> jax.Array:
    """
    x_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[0, 1], [1, 0]])


@jit
def y_operator() -> jax.Array:
    """
    y_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[0, -1j], [1j, 0]])


@jit
def z_operator() -> jax.Array:
    """
    z_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[1, 0], [0, -1]])


@jit
def s_operator() -> jax.Array:
    """
    s_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[1, 0], [0, 1j]])


@jit
def t_operator() -> jax.Array:
    """
    t_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[1, 0], [0, jnp.exp(1j * np.pi / 4)]])


@jit
def controlled_not_operator() -> jax.Array:
    """
    controlled_not_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


@jit
def controlled_z_operator() -> jax.Array:
    """
    controlled_z_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


@jit
def swap_operator() -> jax.Array:
    """
    swap_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


@jit
def sx_operator() -> jax.Array:
    """
    sx_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2


@jit
def controlled_swap_operator() -> jax.Array:
    """
    controlled_swap_operator _summary_

    :return: _description_
    :rtype: jax.Array
    """
    return jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )


@jit
def rx_operator(theta: float) -> jax.Array:
    """
    rx_operator _summary_

    :param theta: _description_
    :type theta: float
    :return: _description_
    :rtype: jax.Array
    """
    term_1 = jnp.cos(theta / 2)
    term_2 = 1j * jnp.sin(-theta / 2)
    return jnp.array([[term_1, term_2], [term_2, term_1]])


@jit
def ry_operator(theta: float) -> jax.Array:
    """
    ry_operator _summary_

    :param theta: _description_
    :type theta: float
    :return: _description_
    :rtype: jax.Array
    """
    term_1 = jnp.cos(theta / 2)
    term_2 = jnp.sin(theta / 2)
    return jnp.array([[term_1, -term_2], [term_2, term_1]])


@jit
def rz_operator(theta: float) -> jax.Array:
    """
    rz_operator _summary_

    :param theta: _description_
    :type theta: float
    :return: _description_
    :rtype: jax.Array
    """
    term1 = jnp.exp(-1j * theta / 2)
    term2 = jnp.exp(1j * theta / 2)
    return jnp.array([[term1, 0], [0, term2]])


@jit
def u3_operator(phi: float, theta: float, omega: float) -> jax.Array:
    """
    u3_operator _summary_

    :param phi: _description_
    :type phi: float
    :param theta: _description_
    :type theta: float
    :param omega: _description_
    :type omega: float
    :return: _description_
    :rtype: jax.Array
    """
    cos_term = jnp.cos(theta / 2)
    sin_term = jnp.sin(theta / 2)
    return jnp.array(
        [
            [cos_term, -jnp.exp(1j * omega) * sin_term],
            [
                jnp.exp(1j * phi) * sin_term,
                jnp.exp(1j * (phi + omega)) * cos_term,
            ],
        ]
    )


def annihilation_operator(cutoff: int) -> jnp.ndarray:
    """
    annihilation_operator _summary_

    :param cutoff: _description_
    :type cutoff: int
    :return: _description_
    :rtype: jnp.ndarray
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, cutoff, dtype=np.complex128)), 1)


def creation_operator(cutoff: int) -> jnp.ndarray:
    """
    creation_operator _summary_

    :param cutoff: _description_
    :type cutoff: int
    :return: _description_
    :rtype: jnp.ndarray
    """
    return jnp.conjugate(annihilation_operator(cutoff=cutoff)).T


def number_operator(cutoff: int) -> jnp.ndarray:
    """
    number_operator _summary_

    :param cutoff: _description_
    :type cutoff: int
    :return: _description_
    :rtype: jnp.ndarray
    """
    return jnp.matmul(creation_operator(cutoff), annihilation_operator(cutoff))


def squeezing_operator(cutoff: int, zeta: complex) -> jnp.ndarray:
    """
    squeezing_operator _summary_

    :param cutoff: _description_
    :type cutoff: int
    :param zeta: _description_
    :type zeta: complex
    :return: _description_
    :rtype: jnp.ndarray
    """
    create = creation_operator(cutoff=cutoff)
    destroy = annihilation_operator(cutoff=cutoff)
    operator = 0.5 * (jnp.conj(zeta) * (destroy @ destroy) - zeta * (create @ create))

    return jitted_exp(operator)


def displacement_operator(cutoff: int, alpha: complex) -> jnp.ndarray:
    """
    displacement_operator _summary_

    :param cutoff: _description_
    :type cutoff: int
    :param alpha: _description_
    :type alpha: complex
    :return: _description_
    :rtype: jnp.ndarray
    """
    create = creation_operator(cutoff=cutoff)
    destroy = annihilation_operator(cutoff=cutoff)
    operator = alpha * create - jnp.conj(alpha) * destroy
    return jitted_exp(operator)


def phase_operator(cutoff: int, theta: float) -> jnp.ndarray:
    r"""
    Returns a phase shift operator, given the dimensions

    .. math::
      \hat{R}(\theta) = \sum_{n=0}^{\text{cutoff}-1} e^{1 n \theta }|n\rangle \langle n|

    This operator applies a phase shift to each Fock state |n⟩ proportional to the
    integer n. The phase shift is given by :math:`e^{i n \theta}`, where `theta` is the
    phase shift parameter.


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
    The phase shift operator is unitary and is used to rotate the phase of a quantum
    state in the Fock basis. The diagonal matrix elements are complex exponentials that
    apply a phase proportional to the Fock state number.
    """
    indices = jnp.arange(cutoff)
    phases = jnp.exp(1j * indices * theta)
    return jnp.diag(phases)


# to do: implement beamsplitter here
@jit
def compute_einsum(
    einsum_str: str, *operands: Union[jax.Array, np.ndarray]
) -> jax.Array:
    """
    Computes einsum using the provided einsum_str and matrices
    with the gpu if accessible (jax.numpy).
    Parameters
    ----------
    einsum_str: str
        Einstein Sum String
    operands: Union[jax.Array, np.ndarray]
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
    kraus_operators: List[Union[np.ndarray, jnp.ndarray]],
) -> jnp.ndarray:
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
    return sum(K @ density_matrix @ jnp.conjugate((K)).T for K in kraus_operators)


def kraus_identity_check(operators: List[jnp.ndarray], tol: float = 1e-6) -> bool:
    """
    Check if Kraus operator sum is less or equal to identity,
    thus representing a valid Kraus Channel

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
    # TODO check if we need dim and identity matrix
    # dim = operators[0].shape[0]
    # identity_matrix = jnp.eye(dim)

    kraus_sum = sum(jnp.matmul(jnp.conjugate(K.T), K) for K in operators)
    identity = jnp.eye(kraus_sum.shape[0])  # type: ignore
    is_valid = jnp.all(jnp.real(jnp.linalg.eigvals(identity - kraus_sum)) >= 0)
    return bool(is_valid)


@jit
def normalize_vector(vector: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
    """
    Normalizes the given vector and returns it
    Parameters
    ----------
    vector: Union[jnp.ndarray, np.ndarray]
        Vector which should be normalized

    Returns
    -------
    jnp.ndarray
        Normalized vector state
    """
    trace = jnp.trace(vector)
    return jnp.array(vector / trace)


@jit
def normalize_matrix(vector: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
    """
    Normalizes the given matrix and returns it

    Parameters
    ----------
    vector: Union[jnp.ndarray, np.ndarray]
        Vector which should be normalized

    Returns
    -------
    jnp.ndarray
        Normalized density matrix
    """
    norm = jnp.linalg.norm(vector)
    return vector / norm


def num_quanta_vector(vector: Union[jnp.ndarray, np.ndarray]) -> int:
    """
    Returns highest possible measurement outcome
    Parameters
    ----------
    vector: Union[jnp.ndarray, np.ndarray]
        vector for which the max possible quanta has to be calculated

    Returns
    -------
    int
        Highest possible measure outcome
    """
    non_zero_indices = jnp.nonzero(vector)[0]
    return int(non_zero_indices[-1])


def num_quanta_matrix(matrix: jnp.ndarray) -> jnp.int64:
    """
    Returns highest possible measurement outcome
    Parameters
    ----------
    matrix: Union[jnp.ndarray, np.ndarray]
        matrix for which the max possible quanta has to be calculated
    """
    non_zero_rows = jnp.any(matrix != 0, axis=1)
    non_zero_cols = jnp.any(matrix != 0, axis=0)

    highest_non_zero_index_row = (
        jnp.where(non_zero_rows)[0][-1].item() if jnp.any(non_zero_rows) else None
    )
    highest_non_zero_index_col = (
        jnp.where(non_zero_cols)[0][-1].item() if jnp.any(non_zero_cols) else None
    )
    assert highest_non_zero_index_row is not None  # for mypy but needs to be checked
    assert highest_non_zero_index_col is not None  # for mypy but needs to be checked
    # Determine the overall highest index
    highest_non_zero_index_matrix = max(
        highest_non_zero_index_row, highest_non_zero_index_col
    )
    return highest_non_zero_index_matrix
