"""Tests for unit parsing functionality."""

import pytest

from noisytracking.units import (
    DEFAULT_DIMENSIONS,
    ParsedUnit,
    parse_unit,
)


# Test data: (input_string, expected_terms_parts, expected_to_string)
# Each term part is (symbol, exponent_name, dimension_name)
UNIT_PARSE_TEST_CASES = [
    # Basic unit
    ("m", [("m", "POSITIVE", "length")], "m"),
    # Division
    ("m/s", [("m", "POSITIVE", "length"), ("s", "NEGATIVE", "time")], "m/s"),
    # Caret exponent
    (
        "m^2",
        [("m", "POSITIVE", "length"), ("m", "POSITIVE", "length")],
        "m²",
    ),
    # Superscript exponent
    (
        "m²",
        [("m", "POSITIVE", "length"), ("m", "POSITIVE", "length")],
        "m²",
    ),
    # Complex: km per hour squared
    (
        "km/h^2",
        [
            ("km", "POSITIVE", "length"),
            ("h", "NEGATIVE", "time"),
            ("h", "NEGATIVE", "time"),
        ],
        "km/h²",
    ),
    # Space multiplication
    (
        "rad h",
        [("rad", "POSITIVE", "angle"), ("h", "POSITIVE", "time")],
        "rad·h",
    ),
    # Dot multiplication
    (
        "m·min",
        [("m", "POSITIVE", "length"), ("min", "POSITIVE", "time")],
        "m·min",
    ),
    # Bracket disambiguation
    (
        "[m][ms]",
        [("m", "POSITIVE", "length"), ("ms", "POSITIVE", "time")],
        "m·ms",
    ),
    # Chained division
    (
        "m/s/s",
        [
            ("m", "POSITIVE", "length"),
            ("s", "NEGATIVE", "time"),
            ("s", "NEGATIVE", "time"),
        ],
        "m/s²",
    ),
    # Double-asterisk exponent
    (
        "m**2",
        [("m", "POSITIVE", "length"), ("m", "POSITIVE", "length")],
        "m²",
    ),
    # Negative superscript
    ("m⁻¹", [("m", "NEGATIVE", "length")], "1/m"),
    # Asterisk multiplication
    (
        "rad * h",
        [("rad", "POSITIVE", "angle"), ("h", "POSITIVE", "time")],
        "rad·h",
    ),
    # Mixed exponents with spaces
    (
        "m^2 / s²",
        [
            ("m", "POSITIVE", "length"),
            ("m", "POSITIVE", "length"),
            ("s", "NEGATIVE", "time"),
            ("s", "NEGATIVE", "time"),
        ],
        "m²/s²",
    ),
    # Double-asterisk with spaces
    (
        "m ** 2",
        [("m", "POSITIVE", "length"), ("m", "POSITIVE", "length")],
        "m²",
    ),
    # Brackets with divisions
    (
        "cm m / [ms][h]",
        [
            ("cm", "POSITIVE", "length"),
            ("m", "POSITIVE", "length"),
            ("ms", "NEGATIVE", "time"),
            ("h", "NEGATIVE", "time"),
        ],
        "cm·m/ms·h",
    ),
]


@pytest.mark.parametrize(
    "input_str,expected_parts,expected_str",
    UNIT_PARSE_TEST_CASES,
    ids=[case[0] for case in UNIT_PARSE_TEST_CASES],
)
def test_parse_unit_terms(
    input_str: str,
    expected_parts: list[tuple[str, str, str]],
    expected_str: str,
) -> None:
    """Test that parse_unit produces correct term parts."""
    result = parse_unit(input_str, DEFAULT_DIMENSIONS)
    assert result.terms_parts == expected_parts


@pytest.mark.parametrize(
    "input_str,expected_parts,expected_str",
    UNIT_PARSE_TEST_CASES,
    ids=[case[0] for case in UNIT_PARSE_TEST_CASES],
)
def test_parse_unit_to_string(
    input_str: str,
    expected_parts: list[tuple[str, str, str]],
    expected_str: str,
) -> None:
    """Test that parsed units convert back to expected string representation."""
    result = parse_unit(input_str, DEFAULT_DIMENSIONS)
    assert result.to_string() == expected_str


@pytest.mark.parametrize(
    "input_str",
    [
        "xyz",  # Unknown unit
        "m/",  # Incomplete expression
        "/s",  # Missing numerator unit
        "m^^2",  # Invalid syntax
    ],
    ids=["unknown_unit", "incomplete_division", "missing_numerator", "invalid_exponent"],
)
def test_parse_unit_invalid_input(input_str: str) -> None:
    """Test that invalid unit expressions raise exceptions."""
    with pytest.raises(Exception):
        parse_unit(input_str, DEFAULT_DIMENSIONS)
