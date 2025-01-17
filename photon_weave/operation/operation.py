from typing import Any, List, Optional, Union

import jax.numpy as jnp

from photon_weave.operation.composite_operation import CompositeOperationType
from photon_weave.operation.custom_state_operation import \
    CustomStateOperationType
from photon_weave.operation.fock_operation import FockOperationType
from photon_weave.operation.polarization_operation import \
    PolarizationOperationType
from photon_weave.state.expansion_levels import ExpansionLevel


class Operation:
    """
    Represents a quantum operation applied to a state in various frameworks (Fock,
    Polarization, Composite, or Custom). The `Operation` class provides functionalities
    for applying quantum operations, managing operator matrices, and computing
    dimensions for these operations. It is designed to handle custom and predefined
    operations, updating internal parameters based on the specific operation type.

    Attributes
    ----------
    _operator : Optional[jnp.ndarray]
        The matrix representation of the operation. This is only used for custom
        operations.
    _operation_type : Union[FockOperationType, PolarizationOperationType,
    CompositeOperationType, CustomStateOperationType]
        The type of the operation, which determines how the operation behaves.
    _apply_count : int
        Specifies how many times this operation will be applied. Defaults to 1.
    _renormalize : bool
        Determines if renormalization is required after applying the operation.
    kwargs : dict
        Additional keyword arguments required for the operation, such as specific
        parameters related to the operation type.
    _expansion_level : ExpansionLevel
        Expansion level required for the operation (inferred from the operation type).
    _expression : str
        String-based expression defining the operation if applicable (e.g., custom
        operations).
    _dimensions : List[int]
        The estimated dimensions required to apply the operation, computed based on the
        number of quanta and state.

    Methods
    -------
    __init__(operation_type, expression=None, apply_count=1, **kwargs)
        Initializes the operation, updates internal parameters based on the operation
        type, and validates required parameters.

    __repr__()
        Provides a string representation of the operation, including matrix formatting
        for custom operators.

    dimensions
        Getter and setter for the dimensions required for the operation.

    compute_dimensions(num_quanta, state)
        Computes the required dimensions for applying the operation based on the
        provided quanta and state.

    required_expansion_level
        Returns the required expansion level for the operation.

    renormalize
        Returns whether renormalization is required after applying the operation.

    operator
        Getter and setter for the operator matrix. This is computed automatically
        unless the operation type is custom.

    Raises
    ------
    KeyError
        If required parameters for the operation type are not provided.
    ValueError
        If an attempt is made to manually set the operator for non-custom operation
        types.
    AssertionError
        If the provided operator is not a valid `jnp.ndarray` for custom operations.
    """

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
        operation_type: Union[
            FockOperationType,
            PolarizationOperationType,
            CompositeOperationType,
            CustomStateOperationType,
        ],
        expression: Optional[str] = None,
        apply_count: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an `Operation` instance with the specified type, expression, and
        application count.

        Parameters
        ----------
        operation_type : Union[FockOperationType, PolarizationOperationType,
        CompositeOperationType, CustomStateOperationType] The type of operation
        (e.g., Fock, Polarization, Composite, or Custom).
        expression : Optional[str], optional
            A string expression for custom operations, by default None.
        apply_count : int, optional
            The number of times to apply the operation, by default 1.
        **kwargs : Any
            Additional keyword arguments required for the operation type.

        Raises
        ------
        KeyError
            If any required parameter for the operation type is missing.
        AssertionError
            If the operator for a custom operation is not a valid `jnp.ndarray`.
        """
        self._operation_type: Union[
            FockOperationType,
            PolarizationOperationType,
            CompositeOperationType,
            CustomStateOperationType,
        ] = operation_type
        self._operator: Optional[jnp.ndarray] = None
        self._apply_count: int = apply_count
        self._renormalize: bool
        self.kwargs = kwargs

        self._operation_type.update(**kwargs)

        if operation_type is FockOperationType.Custom:
            assert isinstance(kwargs["operator"], jnp.ndarray)
            self._operator = kwargs["operator"]
            self._dimensions: List[int] = [self._operator.shape[0]]

        for param in operation_type.required_params:
            if param not in kwargs:
                raise KeyError(
                    f"The '{param}' argument is required for {operation_type.name}"
                )

    def __repr__(self) -> str:
        """
        Returns a string representation of the operation, including the operator matrix
        for custom operations.
        """
        if self._operator is None:
            repr_string = (
                f"{self._operation_type.__class__.__name__}.{self._operation_type.name}"
            )
        else:
            repr_string = f"{self._operation_type.__class__.__name__}"
            repr_string += f".{self._operation_type.name}\n"
            formatted_matrix: Union[str, List[str]]
            formatted_matrix = ""

            if self._operator is None:
                return repr_string

            for row in self._operator:
                formatted_row = "⎢ "  # Start each row with the ⎢ symbol
                for num in row:
                    formatted_row += f"{num.real:+.2f} "  # Include a space
                    # after the real part
                    # Add either "+" or "-" for the imaginary part based on the sign
                    if num.imag >= 0:
                        formatted_row += "+ "
                    else:
                        formatted_row += "- "

                    # Format the imaginary part and add "j"
                    formatted_row += f"{abs(num.imag):.2f}j   "

                formatted_row = formatted_row.strip() + " ⎥\n"
                formatted_matrix += formatted_row

            # Add top and bottom brackets
            formatted_matrix = formatted_matrix.strip().split("\n")
            formatted_matrix[0] = "⎡" + formatted_matrix[0][1:-1] + "⎤"
            formatted_matrix[-1] = "⎣" + formatted_matrix[-1][1:-1] + "⎦"
            formatted_matrix = "\n".join(formatted_matrix)

            repr_string = repr_string + formatted_matrix

        return repr_string

    @property
    def dimensions(self) -> List[int]:
        """
        Gets or sets the estimated dimensions required for applying the operation.

        Returns
        -------
        List[int]
            The list of required dimensions.
        """
        assert isinstance(self._dimensions, list)
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: List[int]) -> None:
        """
        Sets the dimensions required for applying the operation.

        Parameters
        ----------
        dimensions : List[int]
            The dimensions to be set.
        """
        self._dimensions = dimensions

    def compute_dimensions(
        self,
        num_quanta: Union[int, List[int]],
        state: jnp.ndarray,
    ) -> None:
        """
        Computes and updates the required dimensions for applying this operation.

        Parameters
        ----------
        num_quanta : Union[int, List[int]]
            The current maximum number state amplitude or a list of values.
        state : jnp.ndarray
            The traced-out state(s) for dimension estimation.
        """
        if self._operation_type is not FockOperationType.Custom:
            self._dimensions = self._operation_type.compute_dimensions(
                num_quanta, state, **self.kwargs
            )

    @property
    def required_expansion_level(self) -> ExpansionLevel:
        """
        Returns the required expansion level for the operation.

        Returns
        -------
        ExpansionLevel
            The expansion level required.
        """
        return self._operation_type.required_expansion_level

    @property
    def renormalize(self) -> bool:
        """
        Indicates whether renormalization is required after applying the operation.

        Returns
        -------
        bool
            True if renormalization is required, False otherwise.
        """
        return self._operation_type.renormalize

    @property
    def operator(self) -> jnp.ndarray:
        """
        Gets or computes the operator matrix for the operation.

        Returns
        -------
        jnp.ndarray
            The matrix representing the operation.

        Raises
        ------
        AssertionError
            If the computed operator is not a valid `jnp.ndarray`.
        """
        self._operator = self._operation_type.compute_operator(
            self.dimensions, **self.kwargs
        )
        assert isinstance(self._operator, jnp.ndarray)
        return self._operator

    @operator.setter
    def operator(self, operator: jnp.ndarray) -> None:
        """
        Sets the operator matrix for custom operations only.

        Parameters
        ----------
        operator : jnp.ndarray
            The matrix to be set as the operator.

        Raises
        ------
        ValueError
            If the operation type is not custom.
        AssertionError
            If the provided operator is not a valid `jnp.ndarray`.
        """
        assert isinstance(operator, jnp.ndarray)
        if self._operation_type is not FockOperationType.Custom:
            raise ValueError("Operator can only be configured for the Custom types")
        self._operator = operator
