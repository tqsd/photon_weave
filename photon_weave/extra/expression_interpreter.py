import numpy as np
import scipy.linalg as la


def interpreter(expr, context):
    if isinstance(expr, tuple):
        op, *args = expr
        if op == "add":
            result = interpreter(args[0], context)
            for arg in args[1:]:
                result = np.add(result, interpreter(arg, context))
            return result
        elif op == "s_mult":
            result = interpreter(args[0], context)
            for arg in args[1:]:
                result *= interpreter(arg, context)
            return result
        elif op == "m_mult":
            result = interpreter(args[0], context)
            for arg in args[1:]:
                result *= interpreter(arg, context)
            return result
        elif op == "expm":
            return la.expm(interpreter(args[0], context))
        elif op == "div":
            return interpreter(args[0], context) / interpreter(args[1], context)
    elif isinstance(expr, str):
        return context[expr]
    else:
        return expr
