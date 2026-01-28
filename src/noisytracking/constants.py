"""Constants and enums for model dynamics."""

from enum import Enum

CONSTANT_VALUE = 0
CONSTANT_RATE = 1
CONSTANT_CURVATURE = 2

STEADY_STATE = CONSTANT_RATE

CONSTANT_POSITION = CONSTANT_VALUE
CONSTANT_VELOCITY = CONSTANT_RATE
CONSTANT_ACCELERATION = CONSTANT_CURVATURE


class OutlierHandling(str, Enum):
    NONE = "none"
    GATED = "gated"
    HEAVY_TAILED = "heavy_tailed"
    MIXTURE = "mixture"


class ChangeModel(str, Enum):
    DRIFT_WITH_JUMPS = "drift_with_jumps"


class Rel(str, Enum):
    delta = "delta"
