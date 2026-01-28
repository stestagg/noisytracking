from collections import Counter
from enum import Enum
import re
from lark import Lark, Transformer
from lark import Token
from dataclasses import dataclass, field


class Exponent(Enum):
    """Unit exponent: positive (numerator) or negative (denominator)."""
    POSITIVE = 1
    NEGATIVE = -1


@dataclass(unsafe_hash=True)
class Unit:
    name: str
    symbol: str
    factor: float  # Conversion factor to base unit


@dataclass(unsafe_hash=True)
class Dimension:
    name: str
    base_unit: str
    units: frozenset[Unit]


@dataclass
class UnitTerm:
    """A single unit with its exponent, e.g., m^2 or s^-1"""
    symbol: str
    exponent: Exponent
    unit: Unit
    dimension: Dimension
    value: float = 1.0

    @property
    def base_unit(self) -> "UnitTerm":
        """Return a UnitTerm converted to the dimension's base unit with scaled value."""
        # Find the base unit in the dimension
        base = next(u for u in self.dimension.units if u.name == self.dimension.base_unit)
        # Scale the value by the unit's conversion factor
        scaled_value = self.value * self.unit.factor
        return UnitTerm(
            symbol=base.symbol,
            exponent=self.exponent,
            unit=base,
            dimension=self.dimension,
            value=scaled_value,
        )
    
    @property
    def parts(self) -> tuple[str, int, str]:
        """Return (symbol, exponent value, dimension name) tuple for this unit term."""
        return (self.symbol, self.exponent.name, self.dimension.name)


@dataclass
class ParsedUnit:
    """Complete parsed unit expression as list of terms.

    Stored as flat list where exponents are expanded into individual terms.
    Example: m/s^2 → [UnitTerm('m', POSITIVE), UnitTerm('s', NEGATIVE), UnitTerm('s', NEGATIVE)]
    """
    terms: list[UnitTerm] = field(default_factory=list)


    @property
    def terms_parts(self) -> list[tuple[str, int, str]]:
        """Return list of (symbol, exponent, dimension name) tuples for all terms."""
        return [t.parts for t in self.terms]

    def to_string(self) -> str:
        """Format the parsed unit back to a string representation.

        Collapses same-symbol terms with superscript exponents.
        Uses · for multiplication, / for division.

        Examples:
            [m POS, m POS, s NEG, s NEG] → "m²/s²"
            [km POS, h NEG, h NEG] → "km/h²"
            [cm POS, m POS, ms NEG, h NEG] → "cm·m/ms·h"
        """
        # Separate into numerator and denominator
        numerator = [t for t in self.terms if t.exponent == Exponent.POSITIVE]
        denominator = [t for t in self.terms if t.exponent == Exponent.NEGATIVE]

        def collapse_terms(terms: list[UnitTerm]) -> str:
            """Collapse same-symbol terms and format with superscripts."""
            if not terms:
                return ""
            # Count occurrences of each symbol, preserving order
            counts: dict[str, int] = {}
            for t in terms:
                counts[t.symbol] = counts.get(t.symbol, 0) + 1
            # Format each symbol with superscript if count > 1
            parts = []
            for symbol, count in counts.items():
                if count == 1:
                    parts.append(symbol)
                else:
                    parts.append(f"{symbol}{_to_superscript(count)}")
            return "·".join(parts)

        num_str = collapse_terms(numerator)
        den_str = collapse_terms(denominator)

        if den_str:
            return f"{num_str}/{den_str}" if num_str else f"1/{den_str}"
        return num_str or "1"


LENGTH = Dimension(
    name='length',
    base_unit='meter',
    units=frozenset([
        Unit(name='meter', symbol='m', factor=1.0),
        Unit(name='kilometer', symbol='km', factor=1000.0),
        Unit(name='centimeter', symbol='cm', factor=0.01),
    ]),
)
TIME = Dimension(
    name='time',
    base_unit='second',
    units=frozenset([
        Unit(name='millisecond', symbol='ms', factor=0.001),
        Unit(name='second', symbol='s', factor=1.0),
        Unit(name='minute', symbol='min', factor=60.0),
        Unit(name='hour', symbol='h', factor=3600.0),
        Unit(name='day', symbol='d', factor=86400.0),
    ]),
)
ANGLE = Dimension(
    name='angle',
    base_unit='radian',
    units=frozenset([
        Unit(name='radian', symbol='rad', factor=1.0),
        Unit(name='degree', symbol='deg', factor=3.141592653589793 / 180.0),
    ]),
)
OBSERVATION = Dimension(
    name='observation',
    base_unit='observation',
    units=frozenset([
        Unit(name='observation', symbol='obs', factor=1.0),
        Unit(name='observation', symbol='observation', factor=1.0),
        Unit(name='sample', symbol='sample', factor=1.0),
    ]),
)

# Dimensions that can appear alongside any allowed_dimensions (e.g. for velocity, acceleration)
MODIFIER_DIMENSIONS: frozenset[Dimension] = frozenset([TIME, OBSERVATION])


class DimensionValidationError(ValueError):
    """Raised when unit dimensions don't match allowed dimensions."""

    def __init__(
        self,
        unit_string: str,
        expected_dimensions: list[Dimension],
        actual_counts: Counter[Dimension],
        parameter_name: str | None = None,
    ):
        self.unit_string = unit_string
        self.expected_dimensions = expected_dimensions
        self.actual_counts = actual_counts
        self.parameter_name = parameter_name

        expected_counts = Counter(expected_dimensions)
        expected_str = ", ".join(f"{d.name}={c}" for d, c in expected_counts.items())
        actual_str = ", ".join(f"{d.name}={c}" for d, c in actual_counts.items()) or "none"
        param_part = f" for '{parameter_name}'" if parameter_name else ""

        super().__init__(
            f"Invalid dimensions{param_part}: '{unit_string}' "
            f"has [{actual_str}], expected [{expected_str}]"
        )


# Superscript mapping
SUPERSCRIPT_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁻': '-',
}

# Reverse mapping: digits to superscripts
DIGIT_TO_SUPERSCRIPT = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
}


def _to_superscript(n: int) -> str:
    """Convert an integer to Unicode superscript characters."""
    return ''.join(DIGIT_TO_SUPERSCRIPT[c] for c in str(n))


class UnitTransformer(Transformer):
    """Transforms the parse tree into a ParsedUnit structure."""

    def __init__(self, dimensions: list[Dimension]):
        super().__init__()
        # Build symbol -> (Unit, Dimension) lookup map
        self._unit_map: dict[str, tuple[Unit, Dimension]] = {}
        for dim in dimensions:
            for unit in dim.units:
                self._unit_map[unit.symbol] = (unit, dim)

    def _make_unit_term(self, symbol: str, exponent: Exponent) -> UnitTerm:
        """Create a UnitTerm with unit and dimension looked up from the map."""
        unit, dimension = self._unit_map[symbol]
        return UnitTerm(symbol, exponent, unit, dimension)

    def start(self, items):
        return items[0]

    def expr(self, items):
        """Handle division: first term is numerator, rest are denominators."""
        result = ParsedUnit()
        # Filter out DIV tokens, keep only term lists
        terms = [item for item in items if not isinstance(item, Token)]
        for i, term_terms in enumerate(terms):
            for unit_term in term_terms:
                if i == 0:
                    # Numerator - keep exponent as-is
                    result.terms.append(unit_term)
                else:
                    # Denominator - negate exponent
                    negated = Exponent.NEGATIVE if unit_term.exponent == Exponent.POSITIVE else Exponent.POSITIVE
                    result.terms.append(self._make_unit_term(
                        unit_term.symbol, negated
                    ))
        return result

    def term(self, items):
        """Handle multiplication: collect all factors (filter out MUL tokens)."""
        result = []
        for item in items:
            # Skip MUL tokens
            if isinstance(item, Token):
                continue
            # Extend with list of UnitTerms from factor
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def factor(self, items):
        """Handle a unit with optional exponent, expanding to multiple terms."""
        symbol = str(items[0])
        exponent = items[1] if len(items) > 1 else 1
        # Expand exponent into multiple UnitTerm entries with exponent ±1
        if exponent >= 0:
            return [self._make_unit_term(symbol, Exponent.POSITIVE) for _ in range(exponent)]
        else:
            return [self._make_unit_term(symbol, Exponent.NEGATIVE) for _ in range(-exponent)]

    def bracketed_unit(self, items):
        """Handle bracketed unit - just return the symbol string."""
        return str(items[0])

    def exp_caret(self, items):
        """Handle ^N exponent (skip CARET token)."""
        for item in items:
            if not isinstance(item, Token) or item.type == 'SIGNED_INT':
                return int(item)
        return 1

    def exp_dblstar(self, items):
        """Handle **N exponent (skip DBLSTAR token)."""
        for item in items:
            if not isinstance(item, Token) or item.type == 'SIGNED_INT':
                return int(item)
        return 1

    def exp_super(self, items):
        """Handle Unicode superscript exponent."""
        superscript_str = ''.join(str(tok) for tok in items)
        digits = ''.join(SUPERSCRIPT_MAP.get(c, c) for c in superscript_str)
        return int(digits)


def make_grammar(dimensions: list[Dimension]) -> Lark:
    """Build a Lark grammar for parsing unit expressions.

    Args:
        dimensions: List of Dimension objects containing unit definitions.

    Returns:
        A Lark parser configured to parse unit expressions.
    """

    # Collect all unit symbols from dimensions
    symbols = []
    for dim in dimensions:
        for unit in dim.units:
            symbols.append(unit.symbol)

    # Sort by length descending to match longest first (e.g., 'min' before 'm')
    symbols.sort(key=lambda s: (-len(s), s))

    # Build unit regex pattern - escape special regex chars
    escaped_symbols = [re.escape(sym) for sym in symbols]
    unit_pattern = '|'.join(escaped_symbols)

    grammar = f'''
start: expr

// Expression is terms separated by division
expr: term (DIV term)*

// Term is factors multiplied together (space, *, ·, or adjacent brackets)
term: factor (MUL? factor)*

// Factor is a unit with optional exponent, optionally bracketed
factor: bracketed_unit exponent?
      | UNIT exponent?

bracketed_unit: "[" UNIT "]"

// Exponent forms - operators allow optional surrounding whitespace
exponent: CARET SIGNED_INT    -> exp_caret
        | DBLSTAR SIGNED_INT  -> exp_dblstar
        | SUPERSCRIPT+        -> exp_super

// Unit symbols - dynamically generated from Dimensions (as a terminal)
UNIT: /{unit_pattern}/

// Operators - DBLSTAR and CARET must be defined before MUL to get higher priority
// All operators allow optional surrounding whitespace
DBLSTAR: /[ \\t]*\\*\\*[ \\t]*/
CARET: /[ \\t]*\\^[ \\t]*/
DIV: /[ \\t]*\\/[ \\t]*/
// MUL handles: asterisk (not **), middle dot, or whitespace-only (for implicit multiplication)
// Whitespace-only MUL must NOT be followed by /, ^, or * (those belong to other operators)
MUL: /[ \\t]*\\*(?!\\*)[ \\t]*/ | /[ \\t]*·[ \\t]*/ | /[ \\t]+(?![\\/\\^\\*])/

// Unicode superscripts (including minus)
SUPERSCRIPT: /[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]/

SIGNED_INT: /[+-]?[0-9]+/
'''

    return Lark(grammar, start='start', parser='lalr')


DEFAULT_DIMENSIONS = frozenset([LENGTH, TIME, ANGLE, OBSERVATION])

_PARSERS = {}

class UnitParser:

    def __init__(self, dimensions: frozenset[Dimension]=DEFAULT_DIMENSIONS):
        self.dimensions = dimensions
        if dimensions not in _PARSERS:
            _PARSERS[dimensions] = make_grammar(list(dimensions))
        self._parser = _PARSERS[dimensions]
        self._transformer = UnitTransformer(list(dimensions))

    def parse(self, text: str) -> ParsedUnit:
        tree = self._parser.parse(text)
        return self._transformer.transform(tree)


def parse_unit(text: str, dimensions: frozenset[Dimension]=DEFAULT_DIMENSIONS) -> ParsedUnit:
    parser = UnitParser(dimensions)
    return parser.parse(text)


METERS = parse_unit("m")
RADS = parse_unit("rad")


if __name__ == "__main__":
    from tabulate import tabulate

    # Test inputs from verification
    test_inputs = [
        "m",        # basic unit
        "m/s",      # division
        "m^2",      # caret exponent
        "m²",       # superscript exponent
        "km/h^2",   # complex
        "rad h",    # space multiplication
        "m·min",    # dot multiplication
        "[m][ms]",  # bracket disambiguation
        "m/s/s",    # chained division
        "m**2",     # double-asterisk exponent
        "m⁻¹",      # negative superscript
        "rad * h",  # asterisk multiplication
        'm^2 / s²',  # mixed, with spaces
        'm ** 2',   # double-asterisk with spaces
        'cm m / [ms][h]', # brackets with divisions
    ]

    print("Testing unit parser:\n")

    def format_term(t: UnitTerm) -> str:
        """Format a UnitTerm concisely for display."""
        return f"({t.symbol}, {t.exponent.value:+d}, {t.dimension.name})"

    rows = []
    for inp in test_inputs:
        try:
            result = parse_unit(inp)
            formatted = ", ".join(f"({d}:{u} {x[:3]})" for u,x,d in result.terms_parts)
            rows.append((inp, formatted, result.to_string()))
        except Exception as e:
            rows.append((inp, f"ERROR: {e}", ""))

    print(tabulate(rows, headers=["Example", "Result", "to_string()"], tablefmt="rounded_grid"))