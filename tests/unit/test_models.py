"""Tests for core data models."""

from __future__ import annotations

import json
import math
from datetime import datetime

import pytest
from pydantic import ValidationError

from physics_reasoning.core.enums import (
    Difficulty,
    ErrorSeverity,
    ErrorType,
    ProblemSource,
    QuantityRole,
)
from physics_reasoning.core.models import (
    Dimension,
    DimensionCheckResult,
    Equation,
    ExperimentConfig,
    ExperimentResult,
    LLMParsedOutput,
    LLMResponse,
    ParsedEquation,
    ParsedQuantity,
    PhysicsQuantity,
    Problem,
    Solution,
    SolveResult,
    SolveStep,
    ToolCallRecord,
    UnitConversionResult,
    VerificationError,
    VerificationResult,
)


# ── Dimension ─────────────────────────────────────────────────────────────


class TestDimension:
    def test_create_dimensionless(self):
        d = Dimension()
        assert d.is_dimensionless()
        assert d.to_string() == "dimensionless"

    def test_create_force_dimension(self):
        d = Dimension(M=1, L=1, T=-2)
        assert not d.is_dimensionless()
        assert d.to_string() == "M L T^-2"

    def test_from_string_force(self):
        d = Dimension.from_string("M L T^-2")
        assert d.M == 1
        assert d.L == 1
        assert d.T == -2

    def test_from_string_dimensionless(self):
        d = Dimension.from_string("dimensionless")
        assert d.is_dimensionless()

    def test_from_string_empty(self):
        d = Dimension.from_string("")
        assert d.is_dimensionless()

    def test_from_string_with_braces(self):
        d = Dimension.from_string("M L T^{-2}")
        assert d.T == -2

    def test_multiply(self):
        mass = Dimension(M=1)
        acceleration = Dimension(L=1, T=-2)
        force = mass.multiply(acceleration)
        assert force == Dimension(M=1, L=1, T=-2)

    def test_divide(self):
        force = Dimension(M=1, L=1, T=-2)
        mass = Dimension(M=1)
        acceleration = force.divide(mass)
        assert acceleration == Dimension(L=1, T=-2)

    def test_power(self):
        velocity = Dimension(L=1, T=-1)
        v_squared = velocity.power(2)
        assert v_squared == Dimension(L=2, T=-2)

    def test_is_compatible_same(self):
        d1 = Dimension(M=1, L=1, T=-2)
        d2 = Dimension(M=1, L=1, T=-2)
        assert d1.is_compatible(d2)

    def test_is_compatible_different(self):
        force = Dimension(M=1, L=1, T=-2)
        energy = Dimension(M=1, L=2, T=-2)
        assert not force.is_compatible(energy)

    def test_roundtrip_string(self):
        original = Dimension(M=1, L=1, T=-2)
        reconstructed = Dimension.from_string(original.to_string())
        assert original == reconstructed

    def test_equality(self):
        d1 = Dimension(M=1, L=0, T=-2)
        d2 = Dimension(M=1, T=-2)
        assert d1 == d2

    def test_from_string_single_dimension(self):
        d = Dimension.from_string("M")
        assert d == Dimension(M=1)

    def test_from_string_theta(self):
        d = Dimension.from_string("Theta")
        assert d == Dimension(Theta=1)


# ── PhysicsQuantity ───────────────────────────────────────────────────────


class TestPhysicsQuantity:
    def test_create_quantity(self, force_quantity):
        assert force_quantity.name == "force"
        assert force_quantity.symbol == "F"
        assert force_quantity.value == 10.0
        assert force_quantity.unit == "N"

    def test_create_target_quantity(self, acceleration_quantity):
        assert acceleration_quantity.is_target
        assert acceleration_quantity.value is None

    def test_empty_symbol_rejected(self):
        with pytest.raises(ValidationError):
            PhysicsQuantity(name="force", symbol="", dimension="M L T^-2")

    def test_serialization_roundtrip(self, force_quantity):
        data = force_quantity.model_dump()
        reconstructed = PhysicsQuantity.model_validate(data)
        assert reconstructed == force_quantity

    def test_json_roundtrip(self, force_quantity):
        json_str = force_quantity.model_dump_json()
        reconstructed = PhysicsQuantity.model_validate_json(json_str)
        assert reconstructed == force_quantity


# ── Equation ──────────────────────────────────────────────────────────────


class TestEquation:
    def test_create_equation(self, newton2_equation):
        assert newton2_equation.id == "newton2"
        assert "F" in newton2_equation.variables
        assert newton2_equation.domain == "mechanics"

    def test_equation_requires_equals(self):
        with pytest.raises(ValidationError):
            Equation(
                id="bad",
                name="Bad Equation",
                expression="F m a",  # No '='
                variables=["F"],
                variable_quantities={"F": "force"},
                domain="mechanics",
                topic="newton_laws",
            )

    def test_serialization(self, newton2_equation):
        data = newton2_equation.model_dump()
        reconstructed = Equation.model_validate(data)
        assert reconstructed.id == "newton2"


# ── Problem ───────────────────────────────────────────────────────────────


class TestProblem:
    def test_create_problem(self, sample_problem):
        assert sample_problem.id == "test_fma_001"
        assert sample_problem.difficulty == Difficulty.EASY
        assert sample_problem.answer_value == 5.0

    def test_empty_problem_text_rejected(self):
        with pytest.raises(ValidationError):
            Problem(
                id="bad",
                problem_text="",
                topic="newton_laws",
                difficulty=Difficulty.EASY,
                source=ProblemSource.MANUAL,
                answer_value=5.0,
                answer_unit="m/s**2",
            )

    def test_json_roundtrip(self, sample_problem):
        json_str = sample_problem.model_dump_json()
        reconstructed = Problem.model_validate_json(json_str)
        assert reconstructed.id == sample_problem.id
        assert reconstructed.answer_value == sample_problem.answer_value


# ── LLMParsedOutput ──────────────────────────────────────────────────────


class TestLLMParsedOutput:
    def test_create_parsed_output(self, sample_llm_output):
        assert len(sample_llm_output.quantities) == 3
        assert len(sample_llm_output.equations) == 1
        assert sample_llm_output.target_variable == "a"

    def test_empty_output_rejected(self):
        with pytest.raises(ValidationError):
            LLMParsedOutput(
                quantities=[],
                equations=[],
            )

    def test_quantities_only_ok(self):
        output = LLMParsedOutput(
            quantities=[
                ParsedQuantity(name="mass", symbol="m", value=2.0)
            ],
        )
        assert len(output.quantities) == 1


# ── SolveResult ───────────────────────────────────────────────────────────


class TestSolveResult:
    def test_create_solve_result(self, sample_solve_result):
        assert sample_solve_result.target_variable == "a"
        assert sample_solve_result.solutions == [5.0]
        assert sample_solve_result.num_solutions == 1

    def test_empty_solutions(self):
        result = SolveResult(target_variable="x", solutions=[])
        assert result.num_solutions == 0
        assert not result.is_numeric

    def test_multiple_solutions(self):
        result = SolveResult(
            target_variable="v",
            solutions=[5.0, -5.0],
            is_numeric=True,
        )
        assert result.num_solutions == 2


# ── VerificationResult ────────────────────────────────────────────────────


class TestVerificationResult:
    def test_create_pass(self, sample_verification_pass):
        assert sample_verification_pass.is_valid
        assert len(sample_verification_pass.errors) == 0
        assert sample_verification_pass.confidence == 1.0

    def test_create_fail(self):
        result = VerificationResult(
            is_valid=False,
            errors=[
                VerificationError(
                    error_type=ErrorType.DIMENSION_MISMATCH,
                    severity=ErrorSeverity.ERROR,
                    message="Dimension mismatch in F = m * t",
                )
            ],
            checks_performed=["dimensional"],
            checks_passed=[],
            confidence=0.0,
        )
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.DIMENSION_MISMATCH


# ── Solution ──────────────────────────────────────────────────────────────


class TestSolution:
    def test_create_solution(self):
        sol = Solution(
            answer_value=5.0,
            answer_unit="m/s**2",
            is_verified=True,
            num_attempts=1,
        )
        assert sol.answer_value == 5.0
        assert sol.is_verified

    def test_failed_solution(self):
        sol = Solution(
            is_verified=False,
            error_message="Solver returned no solutions",
        )
        assert not sol.is_verified
        assert sol.answer_value is None


# ── Other Models ──────────────────────────────────────────────────────────


class TestToolCallRecord:
    def test_create_record(self):
        record = ToolCallRecord(
            tool_name="solve_equation",
            arguments={"equations": ["F = m * a"], "target_variable": "a"},
            result={"solutions": [5.0]},
            success=True,
            duration_ms=15.3,
        )
        assert record.tool_name == "solve_equation"
        assert record.success


class TestUnitConversionResult:
    def test_create_result(self):
        result = UnitConversionResult(
            value=72.0,
            from_unit="km/h",
            to_unit="m/s",
            from_value=72.0,
            to_value=20.0,
        )
        assert math.isclose(result.to_value, 20.0)


class TestDimensionCheckResult:
    def test_consistent(self):
        result = DimensionCheckResult(
            is_consistent=True,
            lhs_dimension="M L T^-2",
            rhs_dimension="M L T^-2",
            message="Dimensions match",
        )
        assert result.is_consistent


class TestExperimentModels:
    def test_experiment_config(self):
        config = ExperimentConfig(
            experiment_name="test_run",
            model_name="mock/test",
        )
        assert config.random_seed == 42
        assert config.max_retries == 3

    def test_experiment_result(self):
        config = ExperimentConfig(
            experiment_name="test_run",
            model_name="mock/test",
        )
        result = ExperimentResult(
            config=config,
            total_problems=100,
            total_correct=75,
        )
        assert result.total_problems == 100


class TestLLMResponse:
    def test_create_response(self):
        resp = LLMResponse(
            content='{"test": true}',
            model="mock/test",
            finish_reason="stop",
            latency_ms=500.0,
        )
        assert resp.content == '{"test": true}'
        assert resp.usage["total_tokens"] == 0
