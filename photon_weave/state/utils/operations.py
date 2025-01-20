from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

import jax.numpy as jnp

import photon_weave.extra.einsum_constructor as ESC

if TYPE_CHECKING:
    from photon_weave.state.base_state import BaseState


def apply_operation_vector(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
) -> jnp.ndarray:
    """
    Applies the operation to the state vector

    Parameters
    ----------
    state_objs: Union[List[BaseState],Tuple[BaseState,...]]
        List of all base state objects which are in the product state
    states: Union[List[BaseState],Tuple[BaseState,...]]
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
    state_objs = list(state_objs)
    target_states = list(target_states)

    operator_shape = jnp.array([s.dimensions for s in target_states])
    dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))

    shape = [s.dimensions for s in state_objs]
    shape.append(1)

    assert product_state.shape == (dims, 1)
    assert operator.shape == (jnp.prod(operator_shape), jnp.prod(operator_shape))

    product_state = product_state.reshape(shape)
    operator = operator.reshape((*operator_shape, *operator_shape))

    einsum_o = ESC.apply_operator_vector(state_objs, target_states)

    product_state = jnp.einsum(einsum_o, operator, product_state)

    product_state = product_state.reshape((-1, 1))
    # operator = operator.reshape((dims, dims))

    operator = operator.reshape([jnp.prod(operator_shape)] * 2)

    return product_state


def apply_operation_matrix(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
) -> jnp.ndarray:
    """
    Applies the operation to the density matrix

    Parameters
    ----------
    state_objs: Union[List[BaseState],Tuple[BaseState,...]]
        List of all base state objects which are in the product state
    states: Union[List[BaseState],Tuple[BaseState,...]]
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
    state_objs = list(state_objs)
    target_states = list(target_states)

    state_objs = list(state_objs)
    target_states = list(target_states)
    operator_shape = jnp.array([s.dimensions for s in target_states])
    dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
    shape = [s.dimensions for s in state_objs] * 2

    assert product_state.shape == (dims, dims)

    assert operator.shape == (jnp.prod(operator_shape), jnp.prod(operator_shape))

    product_state = product_state.reshape(shape)
    operator = operator.reshape((*operator_shape, *operator_shape))

    dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
    shape = [s.dimensions for s in state_objs] * 2
    shape.append(1)

    einsum_o = ESC.apply_operator_matrix(state_objs, target_states)

    product_state = jnp.einsum(einsum_o, operator, product_state, jnp.conj(operator))

    product_state = product_state.reshape((dims, dims))
    operator = operator.reshape((*operator_shape, *operator_shape))
    return product_state


def apply_kraus_vector(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    operators: Union[List[jnp.ndarray], Tuple[jnp.ndarray, ...]],
) -> jnp.ndarray:
    """
    Applies the channel described with Kraus operators to the state vector

    Parameters
    ----------
    state_objs: Union[List[BaseState],Tuple[BaseState,...]]
        List of all base state objects which are in the product state
    states: Union[List[BaseState],Tuple[BaseState,...]]
        List of the base states on which we want to operate
    product_state: jnp.ndarray
        Product stats (state vector)
    operators: jnp.ndarray
        List of Operator matrix

    Returns
    -------
    jnp.ndarray
        Modified state vector according to the operator

    Notes
    -----
    Given product state needs to be reordered, so that the target states
    are grouped together and their index in the tensor (product state)
    reflects their index in the target states reflects their index in the
    operator. Photon Weave handles this automatically when called from
    within the State Container methods.
    """
    assert isinstance(product_state, jnp.ndarray)
    state_objs = list(state_objs)
    target_states = list(target_states)

    operator_shape = jnp.array([s.dimensions for s in target_states])
    dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))

    shape = [s.dimensions for s in state_objs]
    # shape.append(1)

    assert product_state.shape == (dims, 1)
    expected_operator_dims = int(jnp.prod(operator_shape))
    assert all(
        operator.shape == (expected_operator_dims, expected_operator_dims)
        for operator in operators
    )

    product_state = product_state.reshape((*shape, 1))
    # operators = [o.reshape((*operator_shape, *operator_shape)) for o in operators]

    resulting_state = jnp.zeros_like(product_state)

    # Create padding elements
    pre_pad = jnp.array([[1]])
    post_pad = jnp.array([[1]])
    # padding_condition = True
    for state in state_objs:
        if state not in target_states:
            if state_objs.index(state) < state_objs.index(target_states[0]):
                pre_pad = jnp.kron(pre_pad, jnp.eye(state.dimensions))
            elif state_objs.index(state) > state_objs.index(target_states[-1]):
                post_pad = jnp.kron(post_pad, jnp.eye(state.dimensions))

    einsum_o = ESC.apply_operator_vector(state_objs, state_objs)

    for operator in operators:
        operator = jnp.kron(pre_pad, jnp.kron(operator, post_pad))
        # We need to expand the operator to affect the whole space
        operator = operator.reshape((*shape, *shape))
        resulting_state += jnp.einsum(einsum_o, operator, product_state)

    resulting_state = resulting_state.reshape((-1, 1))
    return resulting_state


def apply_kraus_matrix(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    operators: Union[List[jnp.ndarray], Tuple[jnp.ndarray, ...]],
) -> jnp.ndarray:
    """
    Applies the channel described with Kraus operators to the density matrix

    Parameters
    ----------
    state_objs: Union[List[BaseState],Tuple[BaseState,...]]
        List of all base state objects which are in the product state
    states: Union[List[BaseState],Tuple[BaseState,...]]
        List of the base states on which we want to operate
    product_state: jnp.ndarray
        Product stats (state vector)
    operators: Union[List[jnp.ndarray],Tuple[jnp.ndarray,...]]
        List of Operator matrix

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
    state_objs = list(state_objs)
    target_states = list(target_states)

    operator_shape = jnp.array([s.dimensions for s in target_states])
    dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))

    shape = [s.dimensions for s in state_objs]

    assert product_state.shape == (dims, dims)
    expected_operator_dims = int(jnp.prod(operator_shape))
    assert all(
        operator.shape == (expected_operator_dims, expected_operator_dims)
        for operator in operators
    )

    product_state = product_state.reshape((*shape, *shape))

    einsum_o = ESC.apply_operator_matrix(state_objs, state_objs)

    resulting_state = jnp.zeros_like(product_state)

    # Create padding elements
    pre_pad = jnp.array([[1]])
    post_pad = jnp.array([[1]])
    # TODO
    # padding_condition = True
    for state in state_objs:
        if state not in target_states:
            if state_objs.index(state) < state_objs.index(target_states[0]):
                pre_pad = jnp.kron(pre_pad, jnp.eye(state.dimensions))
            elif state_objs.index(state) > state_objs.index(target_states[-1]):
                post_pad = jnp.kron(post_pad, jnp.eye(state.dimensions))

    for operator in operators:
        operator = jnp.kron(pre_pad, jnp.kron(operator, post_pad))
        # We need to expand the operator to affect the whole space
        operator = operator.reshape((*shape, *shape))
        resulting_state += jnp.einsum(
            einsum_o, operator, product_state, jnp.conj(operator)
        )

    resulting_state = resulting_state.reshape((dims, dims))

    return resulting_state
