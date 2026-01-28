"""Model builder interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from .parameter import Parameter
from .time_policy import SequentialBucketsPolicy, TimePolicy, create_time_policy

@dataclass
class SensorDefinition:
    name: str
    parameter: Parameter
    sensor_data_format: Optional[Callable[[Any], Dict[str, Any]]] = None


def get_model_builder_cls(builder_type: str) -> Type["ModelBuilder"]:
    from .builders.base import ModelBuilder
    from .builders.pyro_builder import PyroBuilder

    if builder_type == "pyro":
        return PyroBuilder
    else:
        raise ValueError(f"Unknown builder type: {builder_type}")


class BuildModel:
    def __init__(self, sample_time_policy: TimePolicy) -> None:
        self.time_policy = sample_time_policy
        self._predicted: Dict[str, Parameter] = {}
        self._sensors: Dict[str, SensorDefinition] = {}
        self._learned: Dict[str, Parameter] = {}

    def predicted(self, name: str, kind: Parameter) -> Parameter:
        kind.name = name
        self._predicted[name] = kind
        return kind

    def sensor(
        self,
        name: str,
        kind: Parameter,
        sensor_data_format: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ) -> Parameter:
        kind.name = name
        self._sensors[name] = SensorDefinition(name=name, parameter=kind, sensor_data_format=sensor_data_format)
        return kind

    def learned(self, name: str, kind: Parameter) -> Parameter:
        kind.name = name
        self._learned[name] = kind
        return kind

    def build(self, kind: str="pyro") -> Any:
        builder_cls = get_model_builder_cls(kind)
        builder = builder_cls(self)
        return builder.build()


def setup(sample_time_policy: Optional[TimePolicy]=None) -> BuildModel:
    if sample_time_policy is None:
        sample_time_policy = SequentialBucketsPolicy()
    return BuildModel(sample_time_policy=sample_time_policy)