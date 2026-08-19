"""Tests for dimensional analysis checker."""

from __future__ import annotations

import pytest

from physics_reasoning.units.dimension_checker import DimensionChecker
from physics_reasoning.units.unit_engine import UnitEngine


@pytest.fixture
def dimension_checker() -> DimensionChecker:
    return DimensionChecker(UnitEngine())


class TestDimensionChecker:
    def test_fma_consistent(self, dimension_checker):
        res = dimension_checker.check_equation(
            "F = m * a",
            {"F": "N", "m": "kg", "a": "m/s**2"},
        )
        assert res.is_consistent
        assert res.lhs_dimension == "M L T^-2"
        assert res.rhs_dimension == "M L T^-2"

    def test_fma_inconsistent_wrong_rhs(self, dimension_checker):
        res = dimension_checker.check_equation(
            "F = m * t",  # mass * time != force
            {"F": "N", "m": "kg", "t": "s"},
        )
        assert not res.is_consistent
        assert "mismatch" in res.message.lower()

    def test_kinetic_energy_consistent(self, dimension_checker):
        res = dimension_checker.check_equation(
            "KE = (1/2) * m * v^2",
            {"KE": "J", "m": "kg", "v": "m/s"},
        )
        assert res.is_consistent
        assert res.lhs_dimension == "M L^2 T^-2"
        assert res.rhs_dimension == "M L^2 T^-2"

    def test_velocity_definition_consistent(self, dimension_checker):
        res = dimension_checker.check_equation(
            "v = d / t",
            {"v": "m/s", "d": "m", "t": "s"},
        )
        assert res.is_consistent
        assert res.lhs_dimension == "L T^-1"
        assert res.rhs_dimension == "L T^-1"

    def test_get_expression_dimension(self, dimension_checker):
        dim = dimension_checker.get_expression_dimension("m * a", {"m": "kg", "a": "m/s**2"})
        assert dim == "M L T^-2"
