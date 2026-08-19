"""Tests for unit conversion engine."""

from __future__ import annotations

import math
import pytest

from physics_reasoning.core.exceptions import UnitConversionError, UnitParseError
from physics_reasoning.units.unit_engine import UnitEngine


@pytest.fixture
def unit_engine() -> UnitEngine:
    return UnitEngine()


class TestUnitEngine:
    def test_convert_km_to_m(self, unit_engine):
        res = unit_engine.convert(5.0, "km", "m")
        assert math.isclose(res.to_value, 5000.0)

    def test_convert_g_to_kg(self, unit_engine):
        res = unit_engine.convert(500.0, "g", "kg")
        assert math.isclose(res.to_value, 0.5)

    def test_convert_kmh_to_ms(self, unit_engine):
        res = unit_engine.convert(72.0, "km/h", "m/s")
        assert math.isclose(res.to_value, 20.0)

    def test_convert_min_to_s(self, unit_engine):
        res = unit_engine.convert(2.5, "min", "s")
        assert math.isclose(res.to_value, 150.0)

    def test_convert_kn_to_n(self, unit_engine):
        res = unit_engine.convert(3.0, "kN", "N")
        assert math.isclose(res.to_value, 3000.0)

    def test_to_si(self, unit_engine):
        val, unit = unit_engine.to_si(72.0, "km/h")
        assert math.isclose(val, 20.0)
        assert "meter" in unit or "m" in unit

    def test_are_compatible_same_dimension(self, unit_engine):
        assert unit_engine.are_compatible("km/h", "m/s")
        assert unit_engine.are_compatible("N", "kg * m / s**2")
        assert unit_engine.are_compatible("J", "N * m")

    def test_are_compatible_incompatible(self, unit_engine):
        assert not unit_engine.are_compatible("kg", "m")
        assert not unit_engine.are_compatible("N", "J")

    def test_get_dimension(self, unit_engine):
        dim_str = unit_engine.get_dimension("m/s**2")
        assert dim_str == "L T^-2"

        dim_force = unit_engine.get_dimension("N")
        assert dim_force == "M L T^-2"

    def test_incompatible_conversion_raises(self, unit_engine):
        with pytest.raises(UnitConversionError):
            unit_engine.convert(10.0, "kg", "m")

    def test_invalid_unit_raises(self, unit_engine):
        with pytest.raises(UnitConversionError):
            unit_engine.convert(10.0, "xyz_not_a_unit_123", "m")
