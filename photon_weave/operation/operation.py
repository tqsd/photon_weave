from enum import Enum
from typing import Any, Optional
import jax.numpy as jnp

from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.operation.fock_operation import FockOperationType
from photon_weave.operation.polarization_operation import PolarizationOperationType


class Operation():
    __slots__ = (
        '_operator', '_operation_type', '_apply_count', '_renormalize', 'kwargs',
        '_expansion_level', '_expression', '_dimensions'
        )
    
    def __init__(self, operation_type: Enum,
                 expression:Optional[str]=None,
                 apply_count:int=1,
                 **kwargs:Any
                 ) -> None:
        if expression is None and operation_type is FockOperationType.Expresion:
            raise ValueError(
                f"For Expression operation type expression is required"
            )

        self._operation_type: Enum = operation_type
        self._operator: Optional[jnp.ndarray] = None
        self._apply_count: int = apply_count
        self._renormalize: bool
        self.kwargs = kwargs

        for param in operation_type.required_params:
            if param not in kwargs:
                raise KeyError(
                    f"The '{param}' argument is required for {operaiton_type.name}"
                )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions:int) -> None:
        self._dimensions = dimensions

    def compute_dimensions(self, num_quanta:int) -> None:
        """
        Returns the esitmated required dimensions for the
        application of this operation

        Parameters
        ----------
        num_quanta: int
            Current maximum number state amplitude
        """
        self._dimensions = int(self._operation_type.compute_dimensions(num_quanta))

    @property
    def required_expansion_level(self) -> ExpansionLevel:
        return self._operation_type.required_expansion_level

    @property
    def renormalize(self) -> bool:
        return self._operation_type.renormalize
        
    @property
    def operator(self) -> jnp.ndarray:
        if self._operation_type != FockOperationType.Custom:
            self._operator = self._operation_type.compute_operator(self.dimensions)
        assert isinstance(self._operator, jnp.ndarray)
        return self._operator

    @operator.setter
    def operator(self, operator: jnp.ndarray):
        assert isinstance(operator, jnp.ndarray)
        if not self._operation_type not in (FockOperationType.Custom):
            raise ValueError(
                f"Operator can only be configured for the Custom types"
            )
        self._operator = operator
