import random
import sys

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, value=None):
        if not hasattr(self, '_initialized'):
            self._initialized = True  # Prevents reinitialization
            self._random_seed = random.randint(0, sys.maxsize)

    def set_seed(self, seed: int) -> None:
        """
        For reproducability one can set a seed for random operations
        Parameters
        ----------
        seed: int
            Seed to be used by random processes
        """
        self._random_seed = seed

    @property
    def random_seed(self) -> int:
        return self._random_seed
