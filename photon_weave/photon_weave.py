import random
import sys
from typing import Any

import jax
import jax.numpy as jnp


class Config:
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Config":
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._initialized = True  # Prevents reinitialization
            self._random_seed = random.randint(0, sys.maxsize)
            self._key = jax.random.PRNGKey(self._random_seed)
            self._contractions = True
            self._dynamic_dimensions = False
            self._use_jit = False

    def set_seed(self, seed: int) -> None:
        """
        For reproducability one can set a seed for random operations
        Parameters
        ----------
        seed: int
            Seed to be used by random processes
        """
        self._random_seed = seed
        self._key = jax.random.PRNGKey(seed)

    @property
    def random_seed(self) -> int:
        return self._random_seed

    @property
    def random_key(self) -> jnp.ndarray:
        """
        Splits the current key and returns a new one for random operations
        """
        key, self._key = jax.random.split(self._key)
        return key

    def set_contraction(self, cs: bool) -> None:
        self._contractions = cs

    @property
    def contractions(self) -> bool:
        return self._contractions

    @property
    def dynamic_dimensions(self) -> bool:
        return self._dynamic_dimensions

    @dynamic_dimensions.setter
    def dynamic_dimensions(self, dynamic_dimensions) -> None:
        self._dynamic_dimensions = bool(dynamic_dimensions)

    def set_dynamic_dimensions(self, dynamic_dimensions: bool) -> None:
        self._dynamic_dimensions = bool(dynamic_dimensions)

    @property
    def use_jit(self) -> bool:
        return self._use_jit

    def set_use_jit(self, use_jit: bool) -> None:
        self._use_jit = bool(use_jit)


class Session:
    """
    Lightweight context manager to scope Config settings per run.

    Example:
        with Session(seed=0, use_jit=True):
            ...
    Restores previous Config values on exit so tests/runs stay isolated.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        contractions: bool | None = None,
        dynamic_dimensions: bool | None = None,
        use_jit: bool | None = None,
    ) -> None:
        cfg = Config()
        self._prev = {
            "seed": cfg.random_seed,
            "key": cfg._key,  # type: ignore[attr-defined]
            "contractions": cfg.contractions,
            "dynamic_dimensions": cfg.dynamic_dimensions,
            "use_jit": cfg.use_jit,
        }
        self._seed = seed
        self._contractions = contractions
        self._dynamic_dimensions = dynamic_dimensions
        self._use_jit = use_jit
        self._cfg = cfg

    def __enter__(self) -> "Config":
        if self._seed is not None:
            self._cfg.set_seed(self._seed)
        if self._contractions is not None:
            self._cfg.set_contraction(self._contractions)
        if self._dynamic_dimensions is not None:
            self._cfg.set_dynamic_dimensions(self._dynamic_dimensions)
        if self._use_jit is not None:
            self._cfg.set_use_jit(self._use_jit)
        return self._cfg

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cfg._random_seed = self._prev["seed"]  # type: ignore[attr-defined]
        self._cfg._key = self._prev["key"]  # type: ignore[attr-defined]
        self._cfg.set_contraction(self._prev["contractions"])
        self._cfg.set_dynamic_dimensions(self._prev["dynamic_dimensions"])
        self._cfg.set_use_jit(self._prev["use_jit"])
