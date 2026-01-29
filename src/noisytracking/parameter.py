"""Core parameter types and compound kinds."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, make_dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .constants import OutlierHandling, Rel
from .relationship import Relationship
from .units import (
    Dimension,
    MODIFIER_DIMENSIONS,
    DimensionValidationError,
    ParsedUnit,
    UnitParser,
    LENGTH,
    ANGLE,
)

if TYPE_CHECKING:
    from .op_graph import OpGraphHelper, OpNode


class Parameter:
    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        self.name = name
        self.units = units
        self._children: Dict[str, Parameter] = {}
        self._relationships: List[Relationship] = []

    @classmethod
    def get_prediction_type(cls) -> Any:
        raise NotImplementedError("Subclasses must implement get_prediction_type method.")

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

    def get_custom_ops(
        self,
        helper: "OpGraphHelper",
        root_name: str,
        leaf_path: Tuple[str, ...],
        leaf: Optional["ScalarParameter"],
    ) -> List["OpNode"]:
        return []
    
    def set_standard_deviation(self, stddev: float) -> None:
        for child in self._children.values():
            child.set_standard_deviation(stddev)

    def set_constant(self, value: Any) -> None:
        raise NotImplementedError("Subclasses must implement set_constant method.")

    def clone(self, name: Optional[str] = None) -> "Parameter":
        cloned = self.__class__(name=name or self.name, units=self.units)
        for child_name, child in self._children.items():
            cloned.add_child(child_name, child.clone(name=child.name))
        return cloned


class ScalarParameter(Parameter):

    @dataclass
    class PredictionType:
        value: float
        standard_deviation: float

    def __init__(
        self,
        name: Optional[str] = None,
        units: Optional[str] = None,
        allowed_dimensions: Optional[list[Dimension]] = None,
        standard_deviation: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, units=None)  # Parse below, not raw string
        self.allowed_dimensions = allowed_dimensions
        # These don't get cloned automatically
        self.standard_deviation = standard_deviation
        self.constant_value: Optional[float] = None

        if units is not None:
            parser = UnitParser()
            parsed = parser.parse(units)
            self._validate_dimensions(parsed, units)
            self.units = parsed
        else:
            self.units = None

    def _validate_dimensions(self, parsed: ParsedUnit, unit_string: str) -> None:
        """Validate that parsed unit dimensions match allowed_dimensions."""
        if self.allowed_dimensions is None:
            return  # No restrictions

        expected_counts = Counter(self.allowed_dimensions)
        actual_counts: Counter[Dimension] = Counter()
        for term in parsed.terms:
            if term.dimension not in MODIFIER_DIMENSIONS:
                actual_counts[term.dimension] += 1

        if actual_counts != expected_counts:
            raise DimensionValidationError(
                unit_string=unit_string,
                expected_dimensions=self.allowed_dimensions,
                actual_counts=actual_counts,
                parameter_name=self.name,
            )
        
    def set_standard_deviation(self, stddev: float) -> None:
        self.standard_deviation = stddev

    def set_constant(self, value: float) -> None:
        self.constant_value = value

    def clone(self, name: Optional[str] = None) -> "ScalarParameter":
        return ScalarParameter(
            name=name or self.name,
            units=self.units.to_string() if self.units else None,
            allowed_dimensions=self.allowed_dimensions,
        )

    @classmethod
    def get_prediction_type(cls) -> Any:
        return ScalarParameter.PredictionType

    def is_compatible_with(self, other: Parameter) -> bool:
        return isinstance(other, ScalarParameter) and self.units.dimension == other.units.dimension


class CompoundParameter(Parameter):

    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        super().__init__(name=name, units=units)        

    @classmethod
    def get_prediction_type(cls) -> Any:
        if '_OutputValueType' not in cls.__dict__:
            fields = []
            for child_name, child in cls().child_parameters.items():
                fields.append((child_name, child.get_prediction_type()))
            cls._OutputValueType = make_dataclass(f"{cls.__name__}Value", fields)
        return cls._OutputValueType
    
    def set_constant(self, value: Dict[str, Any]) -> None:
        for name, param in self.child_parameters.items():
            if name in value:
                param.set_constant(value[name])


class Position(CompoundParameter):
    def __init__(self, name: Optional[str] = None, units: Optional[str] = None) -> None:
        super().__init__(name=name, units=units)
        self.add_child("x", ScalarParameter(name="x", units=units, allowed_dimensions=[LENGTH]))
        self.add_child("y", ScalarParameter(name="y", units=units, allowed_dimensions=[LENGTH]))
        self.add_child("z", ScalarParameter(name="z", units=units, allowed_dimensions=[LENGTH]))


class Rotation(CompoundParameter):
    def __init__(self, name: Optional[str] = None, units: Optional[str] = None) -> None:
        super().__init__(name=name, units=units)
        self.add_child("yaw", ScalarParameter(name="yaw", units=units, allowed_dimensions=[ANGLE]))
        self.add_child("pitch", ScalarParameter(name="pitch", units=units, allowed_dimensions=[ANGLE]))
        self.add_child("roll", ScalarParameter(name="roll", units=units, allowed_dimensions=[ANGLE]))


class Pose(CompoundParameter):
    def __init__(self, name: Optional[str] = None, units: Optional[Any] = None) -> None:
        super().__init__(name=name, units=units)
        position_units = units.get("length") if isinstance(units, dict) else units
        rotation_units = units.get("angle") if isinstance(units, dict) else units
        self.add_child("position", Position(name="position", units=position_units))
        self.add_child("rotation", Rotation(name="rotation", units=rotation_units))
