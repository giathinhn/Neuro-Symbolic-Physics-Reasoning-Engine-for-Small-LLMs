"""Unit engine wrapping Pint for physical unit conversions and standardization."""

from __future__ import annotations

import re
from typing import Any

import pint
from pint import DimensionalityError, Quantity, UndefinedUnitError

from physics_reasoning.core.exceptions import UnitConversionError, UnitParseError
from physics_reasoning.core.models import Dimension, UnitConversionResult
from physics_reasoning.units.unit_registry import get_unit_registry


class UnitEngine:
    """Unit conversion and parsing engine using Pint."""

    def __init__(self):
        self.ureg = get_unit_registry()

    def _normalize_unit_string(self, unit_str: str) -> str:
        """Normalize common unit notation variants to Pint-compatible format."""
        if not unit_str or not unit_str.strip():
            return "dimensionless"

        s = unit_str.strip()
        # Replace ^ with ** for powers: m/s^2 -> m/s**2
        s = re.sub(r"\^([+-]?\d+)", r"**\1", s)
        # Fix common symbols
        s = s.replace("degC", "celsius").replace("°C", "celsius")
        s = s.replace("degF", "fahrenheit").replace("°F", "fahrenheit")
        return s

    def parse_unit(self, unit_str: str) -> Quantity:
        """Parse a unit string into a Pint Quantity (with magnitude 1).

        Args:
            unit_str: String representation of unit (e.g. 'm/s**2', 'km/h', 'N').

        Returns:
            Pint Quantity object with value 1.

        Raises:
            UnitParseError: If unit string cannot be parsed.
        """
        normalized = self._normalize_unit_string(unit_str)
        try:
            return self.ureg.parse_expression(normalized)
        except Exception as e:
            raise UnitParseError(
                f"Failed to parse unit string '{unit_str}': {e}", unit_string=unit_str
            ) from e

    def convert(
        self, value: float, from_unit: str, to_unit: str
    ) -> UnitConversionResult:
        """Convert a numerical value from one unit to another.

        Args:
            value: Float value in from_unit.
            from_unit: Source unit string.
            to_unit: Target unit string.

        Returns:
            UnitConversionResult with from/to values.

        Raises:
            UnitConversionError: If units are dimensionally incompatible.
            UnitParseError: If either unit string is invalid.
        """
        norm_from = self._normalize_unit_string(from_unit)
        norm_to = self._normalize_unit_string(to_unit)

        try:
            qty = self.ureg.Quantity(value, norm_from)
            converted = qty.to(norm_to)
            return UnitConversionResult(
                value=float(converted.magnitude),
                from_unit=from_unit,
                to_unit=to_unit,
                from_value=float(value),
                to_value=float(converted.magnitude),
            )
        except DimensionalityError as e:
            raise UnitConversionError(
                f"Cannot convert between incompatible units: '{from_unit}' -> '{to_unit}': {e}",
                from_unit=from_unit,
                to_unit=to_unit,
            ) from e
        except Exception as e:
            raise UnitConversionError(
                f"Unit conversion failed from '{from_unit}' to '{to_unit}': {e}",
                from_unit=from_unit,
                to_unit=to_unit,
            ) from e

    def to_si(self, value: float, unit: str) -> tuple[float, str]:
        """Convert a quantity to standard SI base units.

        Args:
            value: Numeric value.
            unit: Unit string.

        Returns:
            (si_value, si_unit_string)
        """
        norm = self._normalize_unit_string(unit)
        try:
            qty = self.ureg.Quantity(value, norm)
            si_qty = qty.to_base_units()
            return float(si_qty.magnitude), str(si_qty.units)
        except Exception as e:
            raise UnitConversionError(
                f"Failed to convert '{value} {unit}' to SI base units: {e}",
                from_unit=unit,
            ) from e

    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units are dimensionally compatible."""
        norm1 = self._normalize_unit_string(unit1)
        norm2 = self._normalize_unit_string(unit2)
        try:
            qty1 = self.ureg.Quantity(1.0, norm1)
            qty2 = self.ureg.Quantity(1.0, norm2)
            return qty1.dimensionality == qty2.dimensionality
        except Exception:
            return False

    def get_dimension_object(self, unit: str) -> Dimension:
        """Get the Dimension model representation for a given unit."""
        norm = self._normalize_unit_string(unit)
        try:
            qty = self.ureg.Quantity(1.0, norm)
            dim_dict = qty.dimensionality
            # Pint dimensional keys: [mass] -> M, [length] -> L, [time] -> T,
            # [current] -> I, [temperature] -> Theta, [substance] -> N, [luminosity] -> J
            return Dimension(
                M=int(dim_dict.get("[mass]", 0)),
                L=int(dim_dict.get("[length]", 0)),
                T=int(dim_dict.get("[time]", 0)),
                I=int(dim_dict.get("[current]", 0)),
                Theta=int(dim_dict.get("[temperature]", 0)),
                N=int(dim_dict.get("[substance]", 0)),
                J=int(dim_dict.get("[luminosity]", 0)),
            )
        except Exception:
            return Dimension()

    def get_dimension(self, unit: str) -> str:
        """Get dimensional string (e.g. 'M L T^-2') for a unit."""
        return self.get_dimension_object(unit).to_string()
