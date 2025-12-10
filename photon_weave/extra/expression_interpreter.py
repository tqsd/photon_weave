from typing import Callable, Dict, List

import jax.numpy as jnp
from jax.scipy.linalg import expm


def interpreter(
    expr: tuple,
    context: Dict[str, Callable[[List[int]], jnp.ndarray]],
    dimensions: List[int],
) -> jnp.ndarray:
    """
    Recursively build an operator from a lisp-style expression, a context,
    and subsystem dimensions.

    Parameters
    ----------
    expr : tuple
        Expression in the form ``("command", arg1, arg2, ...)``.
    context : Dict[str, Callable[[List[int]], jnp.ndarray]]
        Mapping from placeholder names to callables that generate operators
        given dimensions.
    dimensions : List[int]
        Dimensions of the target subsystems (in order).

    Returns
    -------
    jnp.ndarray
        The computed operator.

    Notes
    -----
    Supported commands:

    - ``add``: sums all arguments.
    - ``sub``: subtracts the second argument from the first.
    - ``s_mult``: scalar multiplication.
    - ``m_mult``: matrix multiplication.
    - ``div``: division of two arguments.
    - ``kron``: Kronecker product of all arguments.
    - ``expm``: matrix exponential of a single argument.

    Arguments may be numbers, arrays, or strings referencing entries in
    ``context`` (e.g., ``"n"`` resolved via ``context["n"](dimensions)``). Ensure
    the Kronecker order matches the intended subsystem order.
    """
    if isinstance(expr, tuple):
        op, *args = expr
        if op == "add":
            result = interpreter(args[0], context, dimensions)
            for arg in args[1:]:
                result = jnp.add(result, interpreter(arg, context, dimensions))
            return result
        if op == "sub":
            result = interpreter(args[0], context, dimensions)
            result = jnp.subtract(
                result, interpreter(args[1], context, dimensions)
            )
            return result
        elif op == "s_mult":
            result = interpreter(args[0], context, dimensions)
            for arg in args[1:]:
                result *= interpreter(arg, context, dimensions)
            return result
        elif op == "m_mult":
            result = interpreter(args[0], context, dimensions)
            for arg in args[1:]:
                result @= interpreter(arg, context, dimensions)
            return result
        elif op == "kron":
            result = interpreter(args[0], context, dimensions)
            for arg in args[1:]:
                result = jnp.kron(
                    result, interpreter(arg, context, dimensions)
                )
            return result
        elif op == "expm":
            return expm(interpreter(args[0], context, dimensions))
        elif op == "div":
            return interpreter(args[0], context, dimensions) / interpreter(
                args[1], context, dimensions
            )
    elif isinstance(expr, str):
        # Grab a value from the context
        return context[expr](dimensions)
    else:
        # Grab literal value
        return expr
    raise ValueError(
        "Something went wrong in the expression interpreter!", expr
    )
