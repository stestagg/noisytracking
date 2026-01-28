"""Lightweight scaffolding for the noisytracking API."""

from .constants import (
    CONSTANT_ACCELERATION,
    CONSTANT_CURVATURE,
    CONSTANT_POSITION,
    CONSTANT_RATE,
    CONSTANT_VALUE,
    CONSTANT_VELOCITY,
    STEADY_STATE,
    ChangeModel,
    OutlierHandling,
    Rel,
)
from .learned import LearnedBias
from .builder import Builder, setup
from .parameter import (
    CompoundParameter,
    Parameter,
    Pose,
    Position,
    Rotation,
    ScalarParameter,
)
from .prediction import Motion

__all__ = [
    "CONSTANT_ACCELERATION",
    "CONSTANT_CURVATURE",
    "CONSTANT_POSITION",
    "CONSTANT_RATE",
    "CONSTANT_VALUE",
    "CONSTANT_VELOCITY",
    "STEADY_STATE",
    "ChangeModel",
    "CompoundParameter",
    "LearnedBias",
    "Builder",
    "Motion",
    "OutlierHandling",
    "Parameter",
    "Pose",
    "Position",
    "Rel",
    "Rotation",
    "ScalarParameter",
    "setup",
]
