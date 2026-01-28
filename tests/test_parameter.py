"""Tests for Parameter classes and get_prediction_type method."""

import pytest
from dataclasses import is_dataclass, fields

from noisytracking.parameter import (
    Parameter,
    ScalarParameter,
    CompoundParameter,
    Position,
    Rotation,
    Pose,
)
from noisytracking.units import (
    LENGTH,
    ANGLE,
    DimensionValidationError,
    ParsedUnit,
)


class TestScalarParameterPredictionType:
    """Tests for ScalarParameter.get_prediction_type()."""

    def test_returns_prediction_type_class(self):
        """Test that get_prediction_type returns the PredictionType dataclass."""
        prediction_type = ScalarParameter.get_prediction_type()
        assert prediction_type is ScalarParameter.PredictionType

    def test_prediction_type_is_dataclass(self):
        """Test that PredictionType is a dataclass."""
        prediction_type = ScalarParameter.get_prediction_type()
        assert is_dataclass(prediction_type)

    def test_prediction_type_has_expected_fields(self):
        """Test that PredictionType has value and standard_deviation fields."""
        prediction_type = ScalarParameter.get_prediction_type()
        field_names = {f.name for f in fields(prediction_type)}
        assert field_names == {"value", "standard_deviation"}

    def test_prediction_type_can_be_instantiated(self):
        """Test that PredictionType can be instantiated with values."""
        prediction_type = ScalarParameter.get_prediction_type()
        instance = prediction_type(value=1.5, standard_deviation=0.1)
        assert instance.value == 1.5
        assert instance.standard_deviation == 0.1

    def test_instance_method_returns_same_type(self):
        """Test that calling on an instance returns the same type as class method."""
        scalar = ScalarParameter(name="test")
        assert scalar.get_prediction_type() is ScalarParameter.get_prediction_type()


class TestBaseParameterPredictionType:
    """Tests for base Parameter.get_prediction_type()."""

    def test_raises_not_implemented(self):
        """Test that base Parameter raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            Parameter.get_prediction_type()
        assert "Subclasses must implement" in str(exc_info.value)


class TestPositionPredictionType:
    """Tests for Position.get_prediction_type()."""

    def test_returns_dataclass(self):
        """Test that Position.get_prediction_type returns a dataclass."""
        prediction_type = Position.get_prediction_type()
        assert is_dataclass(prediction_type)

    def test_has_xyz_fields(self):
        """Test that the prediction type has x, y, z fields."""
        prediction_type = Position.get_prediction_type()
        field_names = {f.name for f in fields(prediction_type)}
        assert field_names == {"x", "y", "z"}

    def test_xyz_fields_are_scalar_prediction_types(self):
        """Test that x, y, z fields are ScalarParameter.PredictionType."""
        prediction_type = Position.get_prediction_type()
        for field in fields(prediction_type):
            assert field.type is ScalarParameter.PredictionType

    def test_can_instantiate_nested_structure(self):
        """Test creating a Position prediction with nested scalar values."""
        prediction_type = Position.get_prediction_type()
        scalar_type = ScalarParameter.get_prediction_type()

        position_pred = prediction_type(
            x=scalar_type(value=1.0, standard_deviation=0.1),
            y=scalar_type(value=2.0, standard_deviation=0.2),
            z=scalar_type(value=3.0, standard_deviation=0.3),
        )

        assert position_pred.x.value == 1.0
        assert position_pred.y.value == 2.0
        assert position_pred.z.standard_deviation == 0.3


class TestRotationPredictionType:
    """Tests for Rotation.get_prediction_type()."""

    def test_returns_dataclass(self):
        """Test that Rotation.get_prediction_type returns a dataclass."""
        prediction_type = Rotation.get_prediction_type()
        assert is_dataclass(prediction_type)

    def test_has_yaw_pitch_roll_fields(self):
        """Test that the prediction type has yaw, pitch, roll fields."""
        prediction_type = Rotation.get_prediction_type()
        field_names = {f.name for f in fields(prediction_type)}
        assert field_names == {"yaw", "pitch", "roll"}

    def test_fields_are_scalar_prediction_types(self):
        """Test that yaw, pitch, roll fields are ScalarParameter.PredictionType."""
        prediction_type = Rotation.get_prediction_type()
        for field in fields(prediction_type):
            assert field.type is ScalarParameter.PredictionType


class TestPosePredictionType:
    """Tests for Pose.get_prediction_type()."""

    def test_returns_dataclass(self):
        """Test that Pose.get_prediction_type returns a dataclass."""
        prediction_type = Pose.get_prediction_type()
        assert is_dataclass(prediction_type)

    def test_has_position_and_rotation_fields(self):
        """Test that the prediction type has position and rotation fields."""
        prediction_type = Pose.get_prediction_type()
        field_names = {f.name for f in fields(prediction_type)}
        assert field_names == {"position", "rotation"}

    def test_nested_types_match_component_types(self):
        """Test that position field type matches Position's prediction type."""
        pose_type = Pose.get_prediction_type()
        position_type = Position.get_prediction_type()
        rotation_type = Rotation.get_prediction_type()

        pose_fields = {f.name: f.type for f in fields(pose_type)}
        assert pose_fields["position"] is position_type
        assert pose_fields["rotation"] is rotation_type

    def test_can_instantiate_deeply_nested_structure(self):
        """Test creating a Pose prediction with fully nested values."""
        pose_type = Pose.get_prediction_type()
        position_type = Position.get_prediction_type()
        rotation_type = Rotation.get_prediction_type()
        scalar_type = ScalarParameter.get_prediction_type()

        pose_pred = pose_type(
            position=position_type(
                x=scalar_type(value=1.0, standard_deviation=0.1),
                y=scalar_type(value=2.0, standard_deviation=0.1),
                z=scalar_type(value=3.0, standard_deviation=0.1),
            ),
            rotation=rotation_type(
                yaw=scalar_type(value=0.0, standard_deviation=0.01),
                pitch=scalar_type(value=0.1, standard_deviation=0.01),
                roll=scalar_type(value=0.2, standard_deviation=0.01),
            ),
        )

        assert pose_pred.position.x.value == 1.0
        assert pose_pred.rotation.yaw.value == 0.0
        assert pose_pred.rotation.roll.standard_deviation == 0.01


class TestCompoundParameterPredictionTypeCaching:
    """Tests for CompoundParameter prediction type caching behavior."""

    def test_same_type_returned_on_repeated_calls(self):
        """Test that repeated calls return the same type object."""
        type1 = Position.get_prediction_type()
        type2 = Position.get_prediction_type()
        assert type1 is type2

    def test_different_compound_types_are_distinct(self):
        """Test that different compound parameters have distinct types."""
        position_type = Position.get_prediction_type()
        rotation_type = Rotation.get_prediction_type()
        pose_type = Pose.get_prediction_type()

        assert position_type is not rotation_type
        assert position_type is not pose_type
        assert rotation_type is not pose_type


class TestScalarParameterAllowedDimensions:
    """Tests for ScalarParameter allowed_dimensions instance variable."""

    def test_scalar_default_allowed_dimensions_is_none(self):
        """Test ScalarParameter default allowed_dimensions is None."""
        scalar = ScalarParameter(name="test")
        assert scalar.allowed_dimensions is None

    def test_scalar_with_allowed_dimensions(self):
        """Test ScalarParameter stores allowed_dimensions."""
        scalar = ScalarParameter(name="test", units="m", allowed_dimensions=[LENGTH])
        assert scalar.allowed_dimensions == [LENGTH]

    def test_scalar_with_multiple_allowed_dimensions(self):
        """Test ScalarParameter with area (2 LENGTH dimensions)."""
        scalar = ScalarParameter(name="area", units="m^2", allowed_dimensions=[LENGTH, LENGTH])
        assert scalar.allowed_dimensions == [LENGTH, LENGTH]
        assert scalar.units is not None


class TestScalarParameterAutoParsing:
    """Tests for ScalarParameter auto-parsing of units on construction."""

    def test_units_parsed_to_parsed_unit(self):
        """Test that units string is parsed to ParsedUnit."""
        scalar = ScalarParameter(name="x", units="m")
        assert isinstance(scalar.units, ParsedUnit)
        assert len(scalar.units.terms) == 1
        assert scalar.units.terms[0].dimension == LENGTH

    def test_units_none_stays_none(self):
        """Test that None units stay None."""
        scalar = ScalarParameter(name="x", units=None)
        assert scalar.units is None

    def test_velocity_units_parsed(self):
        """Test that velocity units are parsed correctly."""
        scalar = ScalarParameter(name="velocity", units="m/s", allowed_dimensions=[LENGTH])
        assert isinstance(scalar.units, ParsedUnit)
        assert len(scalar.units.terms) == 2


class TestDimensionValidationOnConstruction:
    """Tests for dimension validation happening in __init__."""

    def test_position_accepts_meters(self):
        """Test Position accepts meters (1 LENGTH)."""
        pos = Position(name="target_pos", units="m")
        assert pos.x.units is not None
        assert len(pos.x.units.terms) == 1

    def test_position_accepts_velocity(self):
        """Test Position accepts m/s (1 LENGTH + TIME modifier)."""
        pos = Position(name="target_pos", units="m/s")
        assert pos.x.units is not None
        assert len(pos.x.units.terms) == 2

    def test_position_accepts_acceleration(self):
        """Test Position accepts m/s^2 (1 LENGTH + TIME² modifier)."""
        pos = Position(name="target_pos", units="m/s^2")
        assert pos.x.units is not None
        assert len(pos.x.units.terms) == 3

    def test_position_rejects_radians(self):
        """Test Position rejects rad (0 LENGTH, expected 1)."""
        with pytest.raises(DimensionValidationError) as exc_info:
            Position(name="target_pos", units="rad")
        assert "length=1" in str(exc_info.value)

    def test_position_rejects_squared_meters(self):
        """Test Position rejects m² (2 LENGTH, expected 1)."""
        with pytest.raises(DimensionValidationError) as exc_info:
            Position(name="target_pos", units="m^2")
        assert "x" in str(exc_info.value)

    def test_rotation_accepts_degrees(self):
        """Test Rotation accepts degrees (1 ANGLE)."""
        rot = Rotation(name="target_rot", units="deg")
        assert rot.yaw.units is not None

    def test_rotation_accepts_radians(self):
        """Test Rotation accepts radians (1 ANGLE)."""
        rot = Rotation(name="target_rot", units="rad")
        assert rot.yaw.units is not None

    def test_rotation_accepts_angular_velocity(self):
        """Test Rotation accepts rad/s (1 ANGLE + TIME modifier)."""
        rot = Rotation(name="target_rot", units="rad/s")
        assert rot.yaw.units is not None

    def test_rotation_rejects_meters(self):
        """Test Rotation rejects m (0 ANGLE, expected 1)."""
        with pytest.raises(DimensionValidationError) as exc_info:
            Rotation(name="target_rot", units="m")
        assert "angle=1" in str(exc_info.value)


class TestScalarParameterClone:
    """Tests for ScalarParameter.clone() preserving allowed_dimensions."""

    def test_clone_preserves_allowed_dimensions(self):
        """Test clone preserves allowed_dimensions."""
        scalar = ScalarParameter(name="x", units="m", allowed_dimensions=[LENGTH])
        cloned = scalar.clone(name="y")
        assert cloned.allowed_dimensions == [LENGTH]
        assert cloned.name == "y"

    def test_clone_preserves_units(self):
        """Test clone preserves parsed units."""
        scalar = ScalarParameter(name="x", units="m/s", allowed_dimensions=[LENGTH])
        cloned = scalar.clone()
        assert cloned.units is not None
        assert len(cloned.units.terms) == 2

    def test_clone_without_units(self):
        """Test clone works when units is None."""
        scalar = ScalarParameter(name="x", units=None, allowed_dimensions=[LENGTH])
        cloned = scalar.clone()
        assert cloned.units is None
        assert cloned.allowed_dimensions == [LENGTH]

    def test_clone_without_allowed_dimensions(self):
        """Test clone works when allowed_dimensions is None."""
        scalar = ScalarParameter(name="x", units="m")
        cloned = scalar.clone()
        assert cloned.allowed_dimensions is None
        assert cloned.units is not None


class TestDimensionValidationError:
    """Tests for DimensionValidationError exception."""

    def test_error_message_includes_parameter_name(self):
        """Test error message includes parameter name when provided."""
        from collections import Counter
        error = DimensionValidationError(
            unit_string="rad",
            expected_dimensions=[LENGTH],
            actual_counts=Counter({ANGLE: 1}),
            parameter_name="my_position",
        )
        assert "my_position" in str(error)
        assert "rad" in str(error)

    def test_error_message_without_parameter_name(self):
        """Test error message works without parameter name."""
        from collections import Counter
        error = DimensionValidationError(
            unit_string="rad",
            expected_dimensions=[LENGTH],
            actual_counts=Counter({ANGLE: 1}),
            parameter_name=None,
        )
        assert "rad" in str(error)
        assert "for '" not in str(error)

    def test_error_stores_attributes(self):
        """Test error stores all attributes for inspection."""
        from collections import Counter
        actual = Counter({ANGLE: 1})
        error = DimensionValidationError(
            unit_string="rad",
            expected_dimensions=[LENGTH],
            actual_counts=actual,
            parameter_name="pos",
        )
        assert error.unit_string == "rad"
        assert error.expected_dimensions == [LENGTH]
        assert error.actual_counts == actual
        assert error.parameter_name == "pos"

    def test_error_message_shows_expected_counts(self):
        """Test error message shows expected dimension counts."""
        from collections import Counter
        error = DimensionValidationError(
            unit_string="m",
            expected_dimensions=[LENGTH, LENGTH],  # Area
            actual_counts=Counter({LENGTH: 1}),
            parameter_name=None,
        )
        # Should mention length=2 for area
        assert "length=2" in str(error)

    def test_error_message_for_no_dimensions(self):
        """Test error message when actual has no non-modifier dimensions."""
        from collections import Counter
        error = DimensionValidationError(
            unit_string="s",
            expected_dimensions=[LENGTH],
            actual_counts=Counter(),  # TIME is modifier, so empty
            parameter_name=None,
        )
        assert "none" in str(error)
