from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import jax.numpy as jnp

from photon_weave.core.ops import kraus_identity_check
from photon_weave.photon_weave import Config
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.interfaces import (
    CompositeEnvelopeLike as CompositeEnvelope,
)
from photon_weave.state.interfaces import EnvelopeLike as Envelope

if TYPE_CHECKING:
    from photon_weave.state.polarization import PolarizationLabel

from .utils.measurements import measure_POVM_matrix
from .utils.operations import apply_kraus_matrix
from .utils.representation import representation_matrix, representation_vector
from .utils.routing import route_operation


def _is_envelope(obj: object) -> bool:
    return hasattr(obj, "fock") and hasattr(obj, "polarization")


def _is_polarization(obj: object) -> bool:
    return obj.__class__.__name__ == "Polarization"


class BaseState(ABC):
    __slots__ = (
        "_uid",
        "_expansion_level",
        "_index",
        "_dimensions",
        "_measured",
        "_composite_envelope",
        "_envelope",
        "state",
    )

    @abstractmethod
    def __init__(self) -> None:
        self._uid: Union[str, UUID] = uuid4()
        self._expansion_level: ExpansionLevel = ExpansionLevel.Label
        self._index: Optional[Union[int, Tuple[int, int]]] = None
        self._dimensions: int = -1
        self._composite_envelope: Optional[Any] = None
        self._envelope: Optional[Any] = None
        self.state: Optional[Union[int, Any, jnp.ndarray]] = None

    @property
    def size(self) -> int:
        """
        Returns the size of the state in bytes

        Returns
        -------
        int
            Size of the state in bytes
        """
        if self.state:
            if isinstance(self.state, jnp.ndarray):
                return self.state.nbytes
            else:
                return sys.getsizeof(self.state)
        return 0

    @property
    def envelope(self) -> Union[None, "Envelope"]:
        return self._envelope

    @envelope.setter
    def envelope(self, envelope: Union[None, "Envelope"]) -> None:
        self._envelope = envelope

    @property
    def measured(self) -> bool:
        return self._measured

    @measured.setter
    def measured(self, measured: bool) -> None:
        self._measured = measured

    @property
    def composite_envelope(self) -> Union[None, "CompositeEnvelope"]:
        return self._composite_envelope

    @composite_envelope.setter
    def composite_envelope(
        self, composite_envelope: Union[None, "CompositeEnvelope"]
    ) -> None:
        self._composite_envelope = composite_envelope

    @property
    def uid(self) -> Union[str, UUID]:
        return self._uid

    @uid.setter
    def uid(self, uid: Union[UUID, str]) -> None:
        self._uid = uid

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: int) -> None:
        self._dimensions = dimensions

    @property
    def expansion_level(self) -> Optional[Union[int, ExpansionLevel]]:
        return self._expansion_level

    @expansion_level.setter
    def expansion_level(
        self, expansion_level: Optional["ExpansionLevel"]
    ) -> None:
        self._expansion_level = expansion_level

    @property
    def index(self) -> Union[None, int, Tuple[int, int]]:
        return self._index

    @index.setter
    def index(self, index: Union[None, int, Tuple[int, int]]) -> None:
        self._index = index

    def __hash__(self) -> int:
        return hash(self.uid)

    def __repr__(self) -> str:
        if self.expansion_level == ExpansionLevel.Label:
            if isinstance(self.state, Enum):
                return f"|{self.state.value}⟩"
            elif isinstance(self.state, int):
                return f"|{self.state}⟩"

        if self.index is not None or self.measured:
            return str(self.uid)
        elif self.expansion_level == ExpansionLevel.Vector:
            # Handle cases where the vector has only one element
            assert isinstance(self.state, jnp.ndarray)
            return representation_vector(self.state)
        elif self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            return representation_matrix(self.state)
        return f"{self.uid}"

    def apply_kraus(
        self,
        operators: List[jnp.ndarray],
        identity_check: bool = True,
    ) -> None:
        """
        Apply Kraus operators to the state.
        State is automatically expanded to the density matrix representation

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.Array]]
            List of the operators
        identity_check: bool
            Signal to check whether or not the operators sum up to identity,
            True by default
        """

        assert isinstance(self.expansion_level, ExpansionLevel)

        if identity_check:
            if not kraus_identity_check(operators):
                raise ValueError("Invalid Kraus Channel")
        while self.expansion_level < ExpansionLevel.Matrix:
            self.expand()

        assert isinstance(self.state, jnp.ndarray)
        self.state = apply_kraus_matrix(
            [self],
            [self],
            self.state,
            operators,
            use_contraction=Config().contractions,
        )

        C = Config()
        if C.contractions:
            self.contract()

    @abstractmethod
    def expand(self) -> None:
        pass

    @abstractmethod
    def _set_measured(self) -> None:
        pass

    @abstractmethod
    def extract(self, index: Union[int, Tuple[int, int]]) -> None:
        pass

    def contract(
        self, final: ExpansionLevel = ExpansionLevel.Label, tol: float = 1e-6
    ) -> None:
        """
        Attempts to contract the representation to the level defined in final argument.

        Parameters
        ----------
        final: ExpansionLevel
            Expected expansion level after contraction
        tol: float
            Tolerance when comparing matrices
        """
        raise NotImplementedError(
            "contract must be implemented by concrete state classes"
        )

    @route_operation()
    def measure_POVM(
        self,
        operators: List[jnp.ndarray],
        destructive: bool = True,
        partial: bool = False,
        key: jnp.ndarray | None = None,
    ) -> Tuple[int, Dict["BaseState", int]]:
        """
        Positive Operation-Valued Measurement

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.Array]]
            List of POVM operators
        destructive: bool
            If True measurement is destructive and removed (True by default)
        partial: bool
            If True only this state gets measured, if False and part of envelope, the
            corresponding polarization is measured aswell (False by default)

        Returns
        -------
        Tuple[int, Dict[BaseState, int]]
            Returns a tuple, first element is POVM measurement result, second element
            is a dictionary with the other potentional measurement outcomes

        Notes
        -----
        Method is decorated with route_operation. If the state is
        contained in the product state, the corresponding operation
        will be executed in the state container, which contains this
        stat.
        """
        assert isinstance(self.expansion_level, ExpansionLevel)
        while self.expansion_level < ExpansionLevel.Matrix:
            self.expand()

        assert isinstance(self.state, jnp.ndarray)
        assert self.state.shape == (self.dimensions, self.dimensions)

        outcome, self.state = measure_POVM_matrix(
            [self], [self], operators, self.state, key
        )

        result: Tuple[int, Dict["BaseState", int]] = (outcome, {})
        if destructive:
            self._set_measured()

        if not partial:
            env = self.envelope
            if env is not None and _is_envelope(env):
                state = (
                    env.fock if _is_polarization(self) else env.polarization
                )
                out = state.measure(destructive=destructive, key=key)
                for k, v in out.items():
                    result[1][k] = v

        C = Config()
        if C.contractions and not destructive:
            self.contract()

        return result

    @route_operation()
    def trace_out(self) -> Union[int, "PolarizationLabel", jnp.ndarray]:
        """
        Returns the traced out state of this base state instance.
        If the instance is in envelope it traces out from there.
        If the instance is in composite envelope then it traces it
        out from there

        Returns
        -------
        Union[int, PolarizationLabel, jnp.ndarray]
            Returns traced out state

        Notes
        -----
        Method is decorated with route_operation. If the state is
        contained in the product state, the corresponding operation
        will be executed in the state container, which contains this
        state.
        """
        assert self.state is not None
        return self.state

    @abstractmethod
    def measure(
        self,
        separate_measurement: bool = False,
        destructive: bool = True,
        key: jnp.ndarray | None = None,
    ) -> Dict["BaseState", int]:
        pass

    @abstractmethod
    def measure_expectation(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Differentiable expectation of measuring this subsystem.

        Returns probabilities and expected post-measurement density
        for the unmeasured remainder (empty in the single-state case).
        """
        pass
