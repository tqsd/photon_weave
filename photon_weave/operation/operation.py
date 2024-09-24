from enum import Enum
from typing import Any, List, Optional, Union

import jax.numpy as jnp

from photon_weave.operation.fock_operation import FockOperationType
from photon_weave.operation.polarization_operation import PolarizationOperationType
from photon_weave.state.expansion_levels import ExpansionLevel


class Operation:
    __slots__ = (
        "_operator",
        "_operation_type",
        "_apply_count",
        "_renormalize",
        "kwargs",
        "_expansion_level",
        "_expression",
        "_dimensions",
    )

    def __init__(
        self,
        operation_type: Enum,
        expression: Optional[str] = None,
        apply_count: int = 1,
        **kwargs: Any,
    ) -> None:
        if expression is None and operation_type is FockOperationType.Expresion:
            raise ValueError(f"For Expression operation type expression is required")

        self._operation_type: Enum = operation_type
        self._operator: Optional[jnp.ndarray] = None
        self._apply_count: int = apply_count
        self._renormalize: bool
        self.kwargs = kwargs

        for param in operation_type.required_params:
            if param not in kwargs:
                raise KeyError(
                    f"The '{param}' argument is required for {operation_type.name}"
                )

    def __repr__(self) -> str:
        if self._operator is None:
            repr_string = (
                f"{self._operation_type.__class__.__name__}.{self._operation_type.name}"
            )
        else:
            repr_string = f"{self._operation_type.__class__.__name__}.{self._operation_type.name}\n"
            formatted_matrix: Union[str, List[str]]
            formatted_matrix = "\n".join(
                [
                    "⎢ "
                    + "   ".join(
                        [
                            f"{num.real:+.2f} {'+' if num.imag >= 0 else '-'} {abs(num.imag):.2f}j"
                            for num in row
                        ]
                    )
                    + " ⎥"
                    for row in self._operator
                ]
            )
            formatted_matrix = formatted_matrix.split("\n")
            formatted_matrix[0] = "⎡" + formatted_matrix[0][1:-1] + "⎤"
            formatted_matrix[-1] = "⎣" + formatted_matrix[-1][1:-1] + "⎦"
            formatted_matrix = "\n".join(formatted_matrix)

            repr_string = repr_string + formatted_matrix

        return repr_string

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def compute_dimensions(self, num_quanta: int, state: jnp.ndarray) -> None:
        """
        Returns the esitmated required dimensions for the
        application of this operation

        Parameters
        ----------
        num_quanta: int
            Current maximum number state amplitude
        """
        self._dimensions = int(
            self._operation_type.compute_dimensions(num_quanta, state, **self.kwargs)
        )

    @property
    def required_expansion_level(self) -> ExpansionLevel:
        return self._operation_type.required_expansion_level

    @property
    def renormalize(self) -> bool:
        return self._operation_type.renormalize

    @property
    def operator(self) -> jnp.ndarray:
        if self._operation_type != FockOperationType.Custom:
            self._operator = self._operation_type.compute_operator(
                self.dimensions, **self.kwargs
            )
        assert isinstance(self._operator, jnp.ndarray)
        return self._operator

    @operator.setter
    def operator(self, operator: jnp.ndarray):
        assert isinstance(operator, jnp.ndarray)
        if not self._operation_type not in (FockOperationType.Custom):
            raise ValueError(f"Operator can only be configured for the Custom types")
        self._operator = operator
