"""Shared test fixtures for the physics reasoning engine."""

from __future__ import annotations

import pytest

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.core.enums import Difficulty, ProblemSource, QuantityRole
from physics_reasoning.core.models import (
    Dimension,
    Equation,
    LLMParsedOutput,
    ParsedEquation,
    ParsedQuantity,
    PhysicsQuantity,
    Problem,
    Solution,
    SolveResult,
    VerificationResult,
)


@pytest.fixture
def default_config() -> PipelineConfig:
    """Default pipeline configuration for testing."""
    return PipelineConfig(
        model_name="mock/test-model",
        max_retries=2,
        timeout_seconds=30.0,
        random_seed=42,
    )


@pytest.fixture
def force_quantity() -> PhysicsQuantity:
    return PhysicsQuantity(
        name="force",
        symbol="F",
        value=10.0,
        unit="N",
        dimension="M L T^-2",
        si_unit="newton",
        is_given=True,
    )


@pytest.fixture
def mass_quantity() -> PhysicsQuantity:
    return PhysicsQuantity(
        name="mass",
        symbol="m",
        value=2.0,
        unit="kg",
        dimension="M",
        si_unit="kilogram",
        is_given=True,
    )


@pytest.fixture
def acceleration_quantity() -> PhysicsQuantity:
    return PhysicsQuantity(
        name="acceleration",
        symbol="a",
        dimension="L T^-2",
        si_unit="meter / second ** 2",
        is_target=True,
    )


@pytest.fixture
def newton2_equation() -> Equation:
    return Equation(
        id="newton2",
        name="Newton's Second Law",
        expression="F = m * a",
        variables=["F", "m", "a"],
        variable_quantities={"F": "force", "m": "mass", "a": "acceleration"},
        domain="mechanics",
        topic="newton_laws",
        description="Net force equals mass times acceleration",
        conditions=["constant mass", "inertial frame"],
        dimension_lhs="M L T^-2",
        dimension_rhs="M L T^-2",
    )


@pytest.fixture
def sample_problem(force_quantity, mass_quantity, acceleration_quantity) -> Problem:
    return Problem(
        id="test_fma_001",
        problem_text="A 2 kg object experiences a force of 10 N. Find its acceleration.",
        topic="newton_laws",
        difficulty=Difficulty.EASY,
        source=ProblemSource.MANUAL,
        given_quantities=[force_quantity, mass_quantity],
        target_quantity=acceleration_quantity,
        required_equations=["newton2"],
        answer_value=5.0,
        answer_unit="m/s**2",
        reasoning_steps=[
            "Apply Newton's second law: F = ma",
            "Solve for a: a = F/m",
            "a = 10/2 = 5.0 m/s²",
        ],
    )


@pytest.fixture
def sample_llm_output() -> LLMParsedOutput:
    return LLMParsedOutput(
        problem_understanding="Find the acceleration of a 2 kg object under 10 N force",
        quantities=[
            ParsedQuantity(name="mass", symbol="m", value=2.0, unit="kg", role=QuantityRole.GIVEN),
            ParsedQuantity(name="force", symbol="F", value=10.0, unit="N", role=QuantityRole.GIVEN),
            ParsedQuantity(name="acceleration", symbol="a", unit="m/s**2", role=QuantityRole.TARGET),
        ],
        equations=[
            ParsedEquation(
                equation_id="newton2",
                expression="F = m * a",
                justification="Newton's second law",
            ),
        ],
        target_variable="a",
        solution_steps=["Apply F = ma", "Solve for a = F/m"],
    )


@pytest.fixture
def sample_solve_result() -> SolveResult:
    return SolveResult(
        target_variable="a",
        solutions=[5.0],
        is_numeric=True,
    )


@pytest.fixture
def sample_verification_pass() -> VerificationResult:
    return VerificationResult(
        is_valid=True,
        errors=[],
        checks_performed=["dimensional", "unit", "arithmetic", "bounds", "substitution"],
        checks_passed=["dimensional", "unit", "arithmetic", "bounds", "substitution"],
        confidence=1.0,
    )
