from typing import Tuple, Union

import jax.numpy as jnp

from photon_weave.state.expansion_levels import ExpansionLevel


def state_expand(
    state: Union[jnp.ndarray, int],
    current_expansion_level: ExpansionLevel,
    dimensions: int,
) -> Tuple[jnp.ndarray, ExpansionLevel]:
    """
    Returns the expanded state representation.
    If currently the state is represented as a label,
    it returns the state vector, if it is repersented
    as a state vector, then it is expanded into density
    matrix

    Parameters
    ----------
    state: Union[int, jnp.ndarray]
        State in the form of Label, State Vector or Density Matrix
    current_expansion_level: ExpansionLevel
        Current Expansion Level
    dimensions: int
        Dimensionality of the system

    Returns
    -------
    Tuple[jnp.ndarray, ExpansionLevel]
        Returns expanded state representation with the new representation
        level

    Notes
    -----
    In the case of Polarization, it can expand from State Vector onward
    """
    assert isinstance(dimensions, int)
    assert isinstance(current_expansion_level, ExpansionLevel)
    if dimensions < 0:
        raise ValueError("Dimensions must be larger than 0")
    new_state: Union[jnp.ndarray]
    match current_expansion_level:
        case ExpansionLevel.Label:
            if not isinstance(state, int):
                raise ValueError("Could not expand state, where label is not int type")
            assert state >= 0
            new_state = jnp.zeros(dimensions, dtype=jnp.complex128)
            new_state = new_state.at[state].set(1)
            new_state = new_state[:, jnp.newaxis]
            new_expansion_level = ExpansionLevel.Vector

        case ExpansionLevel.Vector:
            assert isinstance(state, jnp.ndarray)
            assert state.shape == (dimensions, 1)

            new_state = jnp.outer(state.flatten(), jnp.conj(state.flatten()))
            new_expansion_level = ExpansionLevel.Matrix
        case ExpansionLevel.Matrix:
            assert isinstance(state, jnp.ndarray)
            new_state = state
            new_expansion_level = current_expansion_level
        case _:
            raise ValueError("Something went wrong")
    return new_state, new_expansion_level


def state_contract(
    state: Union[int, jnp.ndarray],
    current_expansion_level: ExpansionLevel,
    tol: float = 1e-6,
) -> Tuple[Union[int, jnp.ndarray], ExpansionLevel, bool]:
    """
    Returns contracted state representation if possible

    Parameters
    ----------
    state: Union[int, jnp.ndarray]
        Current state, which should be contracted
    current_expansion_level: ExpansionLevel
        Current expansion level of the state
    tol: float
        Tolerance, used when contracting from matrix to vector representation

    Returns
    -------
    Tuple[Union[jnp.ndarray,int], ExpansionLevel, bool]
        Returns a tuple of the new state, new representation expantion level
        and success flag. Is process was succesfull, then the success is True

    Notes
    -----
    Needs to be handled carefully in the case of Polarization, because
    Polarization uses Enum when representing Label states
    """

    assert isinstance(current_expansion_level, ExpansionLevel)

    success = False
    new_state = state
    new_expansion_level = current_expansion_level

    match current_expansion_level:
        case ExpansionLevel.Matrix:
            assert isinstance(state, jnp.ndarray)
            assert state.shape[0] == state.shape[1]
            state_squared = jnp.matmul(state, state)
            state_trace = jnp.trace(state_squared)
            if jnp.abs(state_trace - 1) < tol:
                eigenvalues, eigenvectors = jnp.linalg.eigh(state)
                pure_state_index = jnp.argmax(eigenvalues)
                new_state = eigenvectors[:, pure_state_index].reshape(-1, 1)
                # Removing the global phase
                assert isinstance(new_state, jnp.ndarray)
                phase = jnp.exp(-1j * jnp.angle(new_state[0]))
                new_state = new_state * phase
                new_expansion_level = ExpansionLevel.Vector
                success = True

        case ExpansionLevel.Vector:
            assert isinstance(state, jnp.ndarray)
            assert state.shape[1] == 1
            ones = jnp.where(state == 1)[0]
            if ones.size == 1:
                new_state = int(ones[0])
                new_expansion_level = ExpansionLevel.Label
                success = True

    return (new_state, new_expansion_level, success)
