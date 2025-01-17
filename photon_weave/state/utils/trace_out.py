from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

import jax.numpy as jnp

import photon_weave.extra.einsum_constructor as ESC

if TYPE_CHECKING:
    from photon_weave.state.base_state import BaseState


def trace_out_vector(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Traces out the system, returning only the space of given target states

    Parameters
    ----------
    state_objs: Union[List[BaseState],Tuple[BaseState, ...]]
        List of the state objects in the product space
    target_states: Union[List[BaseState],Tuple[BaseState, ...]]
        List of the target states, which should be included
        in the resulting subspace
    product_state: jnp.ndarray
        Product state in the vector form

    Returns
    -------
    jnp.ndarray
        Traced out subspace including only the states given as
        target_states. The tensoring of thes spaces in the
        traced out subspace reflects the order of the spaces
        in the target_states.

    Notes
    -----
        Tensoring order in the product space must reflect the order
        of the states given in the state_objs argument.
    """
    state_objs = list(state_objs)
    target_states = list(target_states)

    total_dimensions = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
    shape_dims = [s.dimensions for s in state_objs]
    shape_dims.append(1)

    assert product_state.shape == (total_dimensions, 1)

    product_state = product_state.reshape(shape_dims)

    einsum = ESC.trace_out_vector(state_objs, target_states)

    trace_out_state = jnp.einsum(einsum, product_state)

    return trace_out_state.reshape((-1, 1))


def trace_out_matrix(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Traces out the system, returning only the space of given target states

    Parameters
    ----------
    state_objs: Union[List[BaseState],Tuple[BaseState, ...]]
        List of the state objects in the product space
    target_states: Union[List[BaseState],Tuple[BaseState, ...]]
        List of the target states, which should be included
        in the resulting subspace
    product_state: jnp.ndarray
        Product state in the vector form

    Returns
    -------
    jnp.ndarray
        Traced out subspace including only the states given as
        target_states. The tensoring of thes spaces in the
        traced out subspace reflects the order of the spaces
        in the target_states.

    Notes
    -----
        Tensoring order in the product space must reflect the order
        of the states given in the state_objs argument.
    """

    state_objs = list(state_objs)
    target_states = list(target_states)

    total_dimensions = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
    shape_dimensions = [s.dimensions for s in state_objs] * 2
    new_dimensions = [jnp.prod(jnp.array([s.dimensions for s in target_states]))] * 2

    assert product_state.shape == (total_dimensions, total_dimensions)

    product_state = product_state.reshape(shape_dimensions)

    einsum = ESC.trace_out_matrix(state_objs, target_states)

    traced_out_state = jnp.einsum(einsum, product_state)

    return traced_out_state.reshape(new_dimensions)
