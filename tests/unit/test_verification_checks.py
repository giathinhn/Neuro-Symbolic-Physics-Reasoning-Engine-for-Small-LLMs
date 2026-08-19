"""Tests for individual verification checks."""

from __future__ import annotations

import pytest

from physics_reasoning.core.enums import ErrorType, QuantityRole
from physics_reasoning.core.models import (
    LLMParsedOutput,
    ParsedEquation,
    ParsedQuantity,
    SolveResult,
)
from physics_reasoning.verifier.checks.arithmetic_check import ArithmeticCheck
from physics_reasoning.verifier.checks.bounds_check import BoundsCheck
from physics_reasoning.verifier.checks.dimensional_check import DimensionalCheck
from physics_reasoning.verifier.checks.equation_validity_check import (
    EquationValidityCheck,
)
from physics_reasoning.verifier.checks.substitution_check import SubstitutionCheck
from physics_reasoning.verifier.checks.unit_check import UnitCheck


class TestVerificationChecks:
    def test_dimensional_check_pass(self):
        check = DimensionalCheck()
        parsed = LLMParsedOutput(
            quantities=[
                ParsedQuantity(name="force", symbol="F", value=10.0, unit="N", role=QuantityRole.GIVEN),
                ParsedQuantity(name="mass", symbol="m", value=2.0, unit="kg", role=QuantityRole.GIVEN),
                ParsedQuantity(name="acceleration", symbol="a", unit="m/s**2", role=QuantityRole.TARGET),
            ],
            equations=[ParsedEquation(expression="F = m * a")],
            target_variable="a",
        )
        errors = check.run(parsed, None, [], {"F": "N", "m": "kg", "a": "m/s**2"}, {"F": 10, "m": 2, "a": 5})
        assert len(errors) == 0

    def test_dimensional_check_fail(self):
        check = DimensionalCheck()
        parsed = LLMParsedOutput(
            quantities=[
                ParsedQuantity(name="force", symbol="F", value=10.0, unit="N"),
                ParsedQuantity(name="mass", symbol="m", value=2.0, unit="kg"),
                ParsedQuantity(name="time", symbol="t", value=5.0, unit="s"),
            ],
            equations=[ParsedEquation(expression="F = m * t")],
        )
        errors = check.run(parsed, None, [], {"F": "N", "m": "kg", "t": "s"}, {"F": 10, "m": 2, "t": 5})
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.DIMENSION_MISMATCH

    def test_unit_check_invalid_unit(self):
        check = UnitCheck()
        parsed = LLMParsedOutput(
            quantities=[
                ParsedQuantity(name="mass", symbol="m", value=2.0, unit="unknown_unit_xyz"),
            ],
        )
        errors = check.run(parsed, None, [], {}, {})
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.UNIT_MISMATCH

    def test_arithmetic_check_no_solution(self):
        check = ArithmeticCheck()
        parsed = LLMParsedOutput(quantities=[ParsedQuantity(name="x", symbol="x")])
        solve_res = SolveResult(target_variable="x", solutions=[], is_numeric=False, warnings=["No solution"])
        errors = check.run(parsed, solve_res, [], {}, {})
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.ARITHMETIC_ERROR

    def test_bounds_check_negative_mass(self):
        check = BoundsCheck()
        parsed = LLMParsedOutput(
            quantities=[ParsedQuantity(name="mass", symbol="m", value=-5.0)]
        )
        errors = check.run(parsed, None, [], {}, {"m": -5.0})
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.IMPOSSIBLE_VALUE

    def test_substitution_check_pass(self):
        check = SubstitutionCheck()
        parsed = LLMParsedOutput(
            quantities=[
                ParsedQuantity(name="force", symbol="F", value=10.0),
                ParsedQuantity(name="mass", symbol="m", value=2.0),
                ParsedQuantity(name="acceleration", symbol="a", value=5.0),
            ],
            equations=[ParsedEquation(expression="F = m * a")],
        )
        errors = check.run(parsed, None, [], {}, {"F": 10.0, "m": 2.0, "a": 5.0})
        assert len(errors) == 0

    def test_substitution_check_fail(self):
        check = SubstitutionCheck()
        parsed = LLMParsedOutput(
            quantities=[
                ParsedQuantity(name="force", symbol="F", value=10.0),
                ParsedQuantity(name="mass", symbol="m", value=2.0),
                ParsedQuantity(name="acceleration", symbol="a", value=99.0),
            ],
            equations=[ParsedEquation(expression="F = m * a")],
        )
        errors = check.run(parsed, None, [], {}, {"F": 10.0, "m": 2.0, "a": 99.0})
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.SUBSTITUTION_ERROR

    def test_equation_validity_syntax_error(self):
        check = EquationValidityCheck()
        parsed = LLMParsedOutput(
            quantities=[ParsedQuantity(name="x", symbol="x")],
            equations=[ParsedEquation(expression="F m a")],  # Missing '='
        )
        errors = check.run(parsed, None, [], {}, {})
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.SYNTAX_ERROR
