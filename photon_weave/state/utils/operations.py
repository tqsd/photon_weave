from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, List, Dict

import photon_weave.extra.einsum_constructor as ESC

if TYPE_CHECKING:
    from photon_weave.state.base_state import BaseState


def apply_operation_vector(state_objs: List[BaseState], target_states: List[BaseState],
                    product_state: jnp.ndarray, operator: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the operation to the state vector

    Parameters
    ----------
    state_objs: List[BaseState]
        List of all base state objects which are in the product state
    states: List[BaseState]
        List of the base states on which we want to operate
    product_state: jnp.ndarray
        Product stats (state vector)
    operator: jnp.ndarray
        Operator matrix

    Returns
    -------
    jnp.ndarray
        Modified state vector according to the operator

    Notes
    -----
    Given product state needs to be reordered, so that the target states
    are grouped toghether and their index in the tensor (product state)
    reflects their index in the target states reflects their index in the
    operator. Photon Weave handles this automatically when called from
    within the State Comtainer methods.
    """
    assert isinstance(product_state, jnp.ndarray)
    assert isinstance(operator, jnp.ndarray)

    operator_shape = jnp.array(
        [s.dimensions for s in target_states]
        )
    dims = jnp.prod(
        jnp.array(
            [s.dimensions for s in state_objs]
        ))

    shape = [s.dimensions for s in state_objs]
    shape.append(1)
    
    assert product_state.shape == (dims,1)
    assert operator.shape == (operator_shape, operator_shape)

    product_state = product_state.reshape(shape)
    operator = operator.reshape((*operator_shape, *operator_shape))

    einsum_o = ESC.apply_operator_vector(state_objs, target_states)

    product_state = jnp.einsum(einsum_o, operator, product_state)

    product_state = product_state.reshape((-1,1))
    #operator = operator.reshape((dims, dims))
    operator = operator.reshape([jnp.prod(operator_shape)]*2)

    return product_state

    
def apply_operation_matrix(state_objs: List[BaseState], target_states: List[BaseState],
                    product_state: jnp.ndarray, operator: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the operation to the density matrix

    Parameters
    ----------
    state_objs: List[BaseState]
        List of all base state objects which are in the product state
    states: List[BaseState]
        List of the base states on which we want to operate
    product_state: jnp.ndarray
        Product stats (state vector)
    operator: jnp.ndarray
        Operator matrix

    Returns
    -------
    jnp.ndarray
        Modified state vector according to the operator

    Notes
    -----
    Given product state needs to be reordered, so that the target states
    are grouped toghether and their index in the tensor (product state)
    reflects their index in the target states reflects their index in the
    operator. Photon Weave handles this automatically when called from
    within the State Comtainer methods.
    """
    assert isinstance(product_state, jnp.ndarray)
    assert isinstance(operator, jnp.ndarray)

    operator_shape = jnp.array(
        [s.dimensions for s in target_states]
        )
    dims = jnp.prod(jnp.array(
        [s.dimensions for s in state_objs]
        ))
    shape = [s.dimensions for s in state_objs]*2

    assert product_state.shape == (dims,dims)

    product_state = product_state.reshape(shape)
    operator = operator.reshape((*operator_shape, *operator_shape))

    dims = jnp.prod(jnp.array(
        [s.dimensions for s in state_objs]
        ))
    shape = [s.dimensions for s in state_objs]*2
    shape.append(1)
    
    einsum_o = ESC.apply_operator_matrix(state_objs, target_states)

    product_state = jnp.einsum(
        einsum_o,
        operator,
        product_state,
        jnp.conj(operator))

    product_state = product_state.reshape((dims, dims))
    operator = operator.reshape((*operator_shape,*operator_shape))
    return product_state

    
