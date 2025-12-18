# flake8: noqa

from .composite_operation import CompositeOperationType  # noqa: F401
from .custom_state_operation import CustomStateOperationType  # noqa: F401
from .fock_operation import FockOperationType  # noqa: F401
from .operation import Operation  # noqa: F401
from .polarization_operation import PolarizationOperationType  # noqa: F401


def operation_to_spec(
    operation: Operation,
    targets: tuple[int, ...],
    rep: str,
    use_contraction: bool | None = None,
):
    """
    Convenience helper to emit an OpSpec without importing the IR module at call sites.
    """
    return operation.to_spec(targets=targets, rep=rep, use_contraction=use_contraction)
