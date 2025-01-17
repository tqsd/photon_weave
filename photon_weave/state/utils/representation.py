from typing import List, Union

import jax.numpy as jnp


def representation_vector(state: jnp.ndarray) -> str:
    """
    Returns the state in a nice string in order to display it

    Parameters
    ----------
    state: jnp.ndarray
        State in the form of the vector

    Returns
    -------
    str:
        String representation of the state vector
    """
    assert isinstance(state, jnp.ndarray)
    assert state.shape[1] == 1

    formatted_vector: Union[str, List[str]]
    formatted_vector = ""
    for row in state:
        formatted_row = "⎢ "
        for num in row:
            # Format real part
            formatted_row += f"{num.real:+.2f} "
            if num.imag >= 0:
                formatted_row += "+ "
            else:
                formatted_row += "- "

            formatted_row += f"{abs(num.imag):.2f}j "

        formatted_row = formatted_row.strip() + " ⎥\n"
        formatted_vector += formatted_row
    formatted_vector = formatted_vector.strip().split("\n")
    formatted_vector[0] = "⎡ " + formatted_vector[0][2:-1] + "⎤"
    formatted_vector[-1] = "⎣ " + formatted_vector[-1][2:-1] + "⎦"
    formatted_vector = "\n".join(formatted_vector)
    return formatted_vector


def representation_matrix(state: jnp.ndarray) -> str:
    """
    Returns the state in a nice string in order to display it

    Parameters
    ----------
    state: jnp.ndarray
        State in the form of a density matrix

    Returns
    -------
    str:
        String representation of the state vector
    """
    assert isinstance(state, jnp.ndarray)
    assert state.shape[0] == state.shape[1]
    formatted_matrix: Union[str, List[str]]
    formatted_matrix = ""

    for row in state:
        formatted_row = "⎢ "
        for num in row:
            formatted_row += f"{num.real:+.2f} "
            if num.imag >= 0:
                formatted_row += "+ "
            else:
                formatted_row += "- "

            # Format the imaginary part and add "j"
            formatted_row += f"{abs(num.imag):.2f}j   "

        formatted_row = formatted_row.strip() + " ⎥\n"
        formatted_matrix += formatted_row

    # Add top and bottom brackets
    formatted_matrix = formatted_matrix.strip().split("\n")
    formatted_matrix[0] = "⎡" + formatted_matrix[0][1:-1] + "⎤"
    formatted_matrix[-1] = "⎣" + formatted_matrix[-1][1:-1] + "⎦"
    formatted_matrix = "\n".join(formatted_matrix)

    return f"{formatted_matrix}"
