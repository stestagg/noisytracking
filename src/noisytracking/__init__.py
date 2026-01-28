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
from .builder import BuildModel, setup
from .parameter import (
    CompoundParameter,
    Parameter,
    Pose,
    Position,
    Rotation,
    ScalarParameter,
)
from .prediction import Motion
from .time_policy import (
    DuplicateValueError,
    SequentialBucketsPolicy,
    StaleTimestampError,
    TimePolicy,
    TimePolicyError,
    create_time_policy,
)

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
    "DuplicateValueError",
    "LearnedBias",
    "BuildModel",
    "Motion",
    "OutlierHandling",
    "Parameter",
    "Pose",
    "Position",
    "Rel",
    "Rotation",
    "ScalarParameter",
    "SequentialBucketsPolicy",
    "StaleTimestampError",
    "TimePolicy",
    "TimePolicyError",
    "create_time_policy",
    "setup",
]
