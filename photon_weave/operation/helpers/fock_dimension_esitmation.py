import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from photon_weave.operation import Operation


class FockDimensions:
    """
    An estimator for fock dimensions, used when Displace, Squeeze or Expression

    It works in the following way:
    1.) It gets some initial esimtation, based on num_quanta and operation type
    2.) Computes the expected amplitudes
    3.) Applies an operator and checks the amplitudes
    4.) If more than threshold percentage of state is contained in the
        resulting state then that dimensions are returned
    5.) If not, dimensions are increased and process runs again until
        Enough state is contained in the post operation state
    """

    def __init__(
        self,
        state: jnp.ndarray,
        operation: "Operation",
        num_quanta: int,
        threshold: float,
    ) -> None:
        self.state = state
        self.dimensions = state.shape[0]
        self.operation = operation
        self.threshold = threshold
        self.num_quanta = num_quanta

    def compute_dimensions(self) -> int:
        """
        Computes the dimensions needed for the current state.
        After the dimensions were determined they are returned.

        Returns
        -------
        int
            Always returns a positive integer, denoting number of dimensions needed
        """
        self._initial_estimate()
        while True:
            n = self._compute_dimensions()
            if n > 0:
                return n
            self._increase_dimensions(5)

    def _initial_estimate(self) -> None:
        """
        Initial estimate for the dimensions, implemented for Displace and Squeeze.
        If other type is tried, then current state dimensionality is used for the
        initial guess.
        """
        from photon_weave.operation.fock_operation import FockOperationType

        if self.operation._operation_type is FockOperationType.Displace:
            cutoff = int(
                self.num_quanta + 3 * jnp.abs(self.operation.kwargs["alpha"]) ** 2
            )
            if cutoff > self.dimensions:
                self._increase_dimensions(amount=int(cutoff) - self.dimensions)
        if self.operation._operation_type is FockOperationType.Squeeze:
            r = jnp.abs(self.operation.kwargs["zeta"])
            en = (2 * self.num_quanta + 1) * jnp.sinh(r) ** 2 + self.num_quanta
            en_int = int(jnp.ceil(en))
            cutoff = int(self.num_quanta + 3 * en_int)
            if cutoff > self.dimensions:
                self._increase_dimensions(amount=cutoff - self.dimensions)

    def _compute_dimensions(self) -> int:
        """
        Computes the dimensions needed with current dimension guess

        Returns
        -------
        int
            Number of dimensions required, if -1 then not enough dimensions
            were guessed. Dimensions should be increased then and this
            method needs to be tried again
        """
        if self.state.shape == (self.dimensions, 1):
            self.operation._dimensions = [self.dimensions]
            operator = self.operation._operation_type.compute_operator(
                [self.dimensions], **self.operation.kwargs
            )
            resulting_state = jnp.dot(operator, self.state)
            cdf: float = 0
            if resulting_state[-1, 0] > (1 - self.threshold) * 1e-3:
                return -1
            for i in range(len(resulting_state)):
                tmp = jnp.abs(resulting_state[i][0]) ** 2
                cdf += float(tmp)
                if cdf >= self.threshold:
                    return i + 3
            return -1
        if self.state.shape == (self.dimensions, self.dimensions):
            self.operation._dimensions = [self.dimensions]
            operator = self.operation._operation_type.compute_operator(
                [self.dimensions], **self.operation.kwargs
            )
            resulting_state = operator @ self.state @ operator.T.conj()
            cdf = 0
            if jnp.abs(resulting_state[-1, -1]) > (1 - self.threshold) * 1e-3:
                return -1
            for i in range(self.dimensions):
                tmp = jnp.abs(resulting_state[i, i])
                cdf += float(tmp)

                if cdf >= self.threshold:
                    return i + 3
            return -1
        return -1

    def _increase_dimensions(self, amount: int = 1) -> None:
        """
        Increases the dimensions of the state for the requested amount

        amount: int
            Amount of increase of the dimensions
        """
        if self.state.shape == (self.dimensions, 1):
            pad = jnp.zeros((amount, 1), dtype=self.state.dtype)
            # Vertically stack the padding with the current state
            self.state = jnp.vstack([self.state, pad])
            # Update the dimensions by the amount added
            self.dimensions += amount
        if self.state.shape == (self.dimensions, self.dimensions):
            pad_rows = jnp.zeros((amount, self.dimensions), dtype=self.state.dtype)
            pad_cols = jnp.zeros(
                (self.dimensions + amount, amount), dtype=self.state.dtype
            )

            self.state = jnp.vstack([self.state, pad_rows])

            self.state = jnp.hstack([self.state, pad_cols])

            self.dimensions += amount
