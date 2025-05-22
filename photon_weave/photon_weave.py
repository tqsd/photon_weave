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
