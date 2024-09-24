from enum import Enum, auto

import numpy as np

from photon_weave._math.ops import _expm
from photon_weave.state.expansion_levels import ExpansionLevel


class PolarizationOperationType(Enum):
    I = auto()
    X = auto()
    Y = auto()
    Z = auto()
    H = auto()
    S = auto()
    T = auto()
    PS = auto()
    RX = auto()
    RY = auto()
    RZ = auto()
    U3 = auto()
    Custom = auto()
