"""Core parameter types and compound kinds."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import OutlierHandling, Rel
from .relationship import Relationship


class Parameter:
    """Base parameter in the model builder tree."""

    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        self.name = name
        self.units = units
        self._children: Dict[str, Parameter] = {}
        self._relationships: List[Relationship] = []

    @property
    def child_parameters(self) -> Dict[str, "Parameter"]:
        return dict(self._children)

    def add_child(self, name: str, parameter: "Parameter") -> None:
        self._children[name] = parameter

    def __getattr__(self, item: str) -> "Parameter":
        if item in self._children:
            return self._children[item]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {item}")

    def is_estimated_from(
        self,
        source: "Parameter",
        rel: Optional[Rel] = None,
        outlier_handling: Optional[OutlierHandling] = None,
    ) -> None:
        self._relationships.append(
            Relationship(source=source, rel=rel, outlier_handling=outlier_handling)
        )

    def clone(self, name: Optional[str] = None) -> "Parameter":
        cloned = self.__class__(name=name or self.name, units=self.units)
        for child_name, child in self._children.items():
            cloned.add_child(child_name, child.clone(name=child.name))
        return cloned


class ScalarParameter(Parameter):
    """A leaf parameter."""

    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        super().__init__(name=name, units=units)


class CompoundParameter(Parameter):
    """Parameter composed of sub-parameters."""

    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        super().__init__(name=name, units=units)


class Position(CompoundParameter):
    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        super().__init__(name=name, units=units)
        self.add_child("x", ScalarParameter(name="x", units=units))
        self.add_child("y", ScalarParameter(name="y", units=units))
        self.add_child("z", ScalarParameter(name="z", units=units))


class Rotation(CompoundParameter):
    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        super().__init__(name=name, units=units)
        self.add_child("yaw", ScalarParameter(name="yaw", units=units))
        self.add_child("pitch", ScalarParameter(name="pitch", units=units))
        self.add_child("roll", ScalarParameter(name="roll", units=units))


class Pose(CompoundParameter):
    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        super().__init__(name=name, units=units)
        position_units = units.get("length") if isinstance(units, dict) else units
        rotation_units = units.get("angle") if isinstance(units, dict) else units
        self.add_child("position", Position(name="position", units=position_units))
        self.add_child("rotation", Rotation(name="rotation", units=rotation_units))