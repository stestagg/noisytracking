"""Model builder interface."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .parameter import Parameter


class Builder:
    def __init__(self, time_field: str, time_units: str, sample_time_policy: str) -> None:
        self.time_field = time_field
        self.time_units = time_units
        self.sample_time_policy = sample_time_policy
        self._predicted: Dict[str, Parameter] = {}
        self._sensors: Dict[str, Parameter] = {}
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
        kind.sensor_data_format = sensor_data_format
        self._sensors[name] = kind
        return kind

    def learned(self, name: str, kind: Parameter) -> Parameter:
        kind.name = name
        self._learned[name] = kind
        return kind

    def build(self) -> Any:
        raise NotImplementedError("Model building is not implemented yet.")


def setup(time_field: str, time_units: str, sample_time_policy: str) -> Builder:
    return Builder(
        time_field=time_field, time_units=time_units, sample_time_policy=sample_time_policy
    )
