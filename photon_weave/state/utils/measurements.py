from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, List, Dict

import photon_weave.extra.einsum_constructor as ESC
from photon_weave.photon_weave import Config

if TYPE_CHECKING:
    from photon_weave.state.base_state import BaseState

def measure_vector(state_objs: List[BaseState], target_states: List[BaseState],
                   product_state: jnp.ndarray) -> Tuple[Dict[BaseState,int], jnp.ndarray]:
    """
    Measures state vector and returns the outcome with the
    post measurement state.

    Parameters
    ----------
    state_objs: List[BaseState]
        List of all state objects which are in the probuct state
    states: List[BaseState]
        List of the states, which should be measured

    Returns
    -------
    Tuple[Dict[BaseState, int], jnp.ndarray]
        A Tuple containing:
        - A dictionary mapping each target state to its measurement
        - A jax array representing the post measuremen state of the rest of the
          product state
    """
    assert isinstance(product_state, jnp.ndarray)
    expected_dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
    assert product_state.shape == (expected_dims,1)

    # Reshape the array into the tensor
    shape = [s.dimensions for s in state_objs]
    shape.append(1)
    product_state = product_state.reshape(shape)

    C = Config()
    outcomes = {}
    for idx, state in enumerate(target_states):
        # Using einsum string we compute the outcome probabilities
        einsum_m = ESC.measure_vector(state_objs, [state])
        projected_state = jnp.einsum(einsum_m, product_state)
        probabilities = jnp.abs(projected_state.flatten())**2
        probabilities /= jnp.sum(probabilities)

        # Based on the probabilities and key we choose outcome
        key = C.random_key
        outcome = jax.random.choice(
            key,
            a=jnp.arange(state.dimensions),
            p=probabilities
            )
        outcomes[state] = int(outcome)
        
        # We remove the measured system from the product state
        indices: List[Union[slice, int]] = [slice(None)] * len(product_state.shape)
        indices[state_objs.index(state)] = outcomes[state]
        product_state = product_state[tuple(indices)]

        state_objs.remove(state)
    

    product_state = product_state.reshape(-1,1)

    return outcomes, product_state
        
    
def measure_matrix(state_objs: List[BaseState], target_states: List[BaseState],
                   product_state: jnp.ndarray) -> Tuple[Dict[BaseState,int], jnp.ndarray]:
    """
    Measures Density Matrix and returns the outcome with the
    post measurement state of the product states, which were not measured.

    Parameters
    ----------
    state_objs: List[BaseState]
        List of all state objects which are in the probuct state
    states: List[BaseState]
        List of the states, which should be measured

    Returns
    -------
    Tuple[Dict[BaseState, int], jnp.ndarray]
        A Tuple containing:
        - A dictionary mapping each target state to its measurement
        - A jax array representing the post measuremen state of the rest of the
          product state
    """
    assert isinstance(product_state, jnp.ndarray)
    expected_dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
    assert product_state.shape == (expected_dims, expected_dims)

    # Reshape the array into the tensor
    shape = [s.dimensions for s in state_objs] * 2
    product_state = product_state.reshape(shape)

    # Assume that the state is correctly reshaped
    # TODO: assert
    C = Config()
    outcomes = {}
    for idx, state in enumerate(target_states):
        # Using einsum string we compute the outcome probabilities
        einsum_m = ESC.measure_matrix(state_objs, [state])
        projected_state = jnp.einsum(einsum_m, product_state)
        
        probabilities = jnp.abs(jnp.diag(projected_state))
        probabilities /= jnp.sum(probabilities)

        # Based on the probabilities and key we choose outcome
        key = C.random_key
        outcome = jax.random.choice(
            key,
            a=jnp.arange(state.dimensions),
            p=probabilities
            )
        outcomes[state] = int(outcome)

        # Construct post measurement state
        state_index = state_objs.index(state)
        row_idx = state_index
        col_idx = state_index + len(state_objs)
        indices = [slice(None)] * len(product_state.shape)
        indices[row_idx] = outcomes[state]
        indices[col_idx] = outcomes[state]
        product_state = product_state[tuple(indices)]

        state_objs.remove(state)

    if len(state_objs) > 0:
        new_dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
        product_state = product_state.reshape((new_dims, new_dims))

    return outcomes, product_state
