from typing import Callable, Dict, List

import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.numpy.linalg import matrix_power


def interpreter(
    expr: tuple,
    context: Dict[str, Callable[[List[int]], jnp.ndarray]],
    dimensions: List[int],
) -> jnp.ndarray:
    r"""
        Recursively compute an operator from an expression, context and dimensions

    Parameters
    ----------
        expr: tuple
            Expression defined in a lisp style tuples
        context: Dict[str, Callable[[List[int]], jnp.ndarray]]
            Context is a dictionary of callables (lambda functions)
            Every lambda should accept list of integers (dimensions)
            and return a jnp.ndarray operator matrix
        dimensions: List[int]
            List of dimensions, ordered in the same way as the states
            operated on

    Returns
    -------
        jnp.ndarray
            Correctly computed operator

    Notes
    -----
        When constructing operator which acts on multiple spaces, be
        carefull that you correctly kron the operators. Wrong order
        will not result in correct operator, but no exceptions will
        be produced.

    Usage
    -----
        Tuples are evaluated from the inner most tuple to the outer most
        one. The first element in a tuple is a `command`, followed by
        `arguments˙. There can be multiple arguments for some commands.
        Interpreter understands the following commands:
            - add: adds all arguments together, accepts any number of arguments
            >>> ('add', 1, 2, 3) -> Sums the arguments and is evaluated to 6
            - sub: substracts the arguments, accepts 2 arguments
            >>> ('sub', 3, 2) -> 1
            - s_mult: Scalar multiplication, accepts any number of arguments
            >>> ('s_mult', 1,2,A) -> 2*A, where A is a matrix
            - m_mult: Matrix multiplication, accepts and number of arguments
            >>> ('m_mult', A,B,C) -> A@B@C
            - div: Divides the values, accpets two arguments
            >>> ('div', A, B) -> A/B
            - kron: Kronecker multiplication, accepts and number of arguments
            >>> ('kron', A,B,C) -> jnp.kron(A,jnp.kron(B,C))
            - expm: Exponentiate matrix term, accepts one argument
            >>> ('expm', A) -> e^A
            - s_pow: Raises a scalar value to the power of another scalar value, accepts two arguments
            >>> ('s_pow', 2, 3) -> 2^3
            - m_pow: Raises a matrix to the power of a scalar, accepts two arguments
            >>> ('m_pow', A, 3) -> A^3
            Expressions can be nested to produce complex expressions:
            >>> ('expm', ('s_mult', 1j, jnp.pi, 'n'))

        Arguments can be numbers, arrays or matrices. Arguments can also
        be string types. If an argument is a string type, like in example 'n',
        then its value will be computed from the context.
            >>> context = {
            >>>    'n': lambda dims: number_operator(dims[0])
            >>>}
            In this case
            'n' will be computed as:
            >>> context['n'](dims)
        The index 0 tells the interpreter the which state this operator should
        correspond to. The list `dims` is passed by the interpreter and it contains, the
        list of dimensions of the state we wish to operate on.

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
            result = jnp.subtract(result, interpreter(args[1], context, dimensions))
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
                result = jnp.kron(result, interpreter(arg, context, dimensions))
            return result
        elif op == "expm":
            return expm(interpreter(args[0], context, dimensions))
        elif op == "div":
            return interpreter(args[0], context, dimensions) / interpreter(
                args[1], context, dimensions
            )
        elif op == "s_pow":
            result = jnp.pow(interpreter(args[0], context, dimensions), interpreter(args[1], context, dimensions))
            return result
        elif op == "m_pow":
            result = matrix_power(interpreter(args[0], context, dimensions), interpreter(args[1], context, dimensions))
            return result
    elif isinstance(expr, str):
        # Grab a value from the context
        return context[expr](dimensions)
    else:
        # Grab literal value
        return expr
    raise ValueError("Something went wrong in the expression interpreter!", expr)
