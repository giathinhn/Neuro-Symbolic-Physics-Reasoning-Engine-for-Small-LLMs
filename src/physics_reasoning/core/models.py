"""Pydantic data models for the physics reasoning engine."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from physics_reasoning.core.enums import (
    Difficulty,
    ErrorSeverity,
    ErrorType,
    ProblemSource,
    QuantityRole,
)


# ---------------------------------------------------------------------------
# Dimension
# ---------------------------------------------------------------------------

class Dimension(BaseModel):
    """Dimensional representation using SI base dimensions.

    Uses integer exponents for each base dimension:
    M (mass), L (length), T (time), I (current),
    Theta (temperature), N (amount), J (luminous intensity).
    """

    M: int = 0
    L: int = 0
    T: int = 0
    I: int = 0
    Theta: int = 0
    N: int = 0
    J: int = 0

    def multiply(self, other: Dimension) -> Dimension:
        """Multiply dimensions (add exponents)."""
        return Dimension(
            M=self.M + other.M,
            L=self.L + other.L,
            T=self.T + other.T,
            I=self.I + other.I,
            Theta=self.Theta + other.Theta,
            N=self.N + other.N,
            J=self.J + other.J,
        )

    def divide(self, other: Dimension) -> Dimension:
        """Divide dimensions (subtract exponents)."""
        return Dimension(
            M=self.M - other.M,
            L=self.L - other.L,
            T=self.T - other.T,
            I=self.I - other.I,
            Theta=self.Theta - other.Theta,
            N=self.N - other.N,
            J=self.J - other.J,
        )

    def power(self, n: int) -> Dimension:
        """Raise dimension to a power (multiply all exponents by n)."""
        return Dimension(
            M=self.M * n,
            L=self.L * n,
            T=self.T * n,
            I=self.I * n,
            Theta=self.Theta * n,
            N=self.N * n,
            J=self.J * n,
        )

    def is_compatible(self, other: Dimension) -> bool:
        """Check if two dimensions are identical."""
        return self == other

    def is_dimensionless(self) -> bool:
        """Check if this is a dimensionless quantity."""
        return all(
            v == 0
            for v in (self.M, self.L, self.T, self.I, self.Theta, self.N, self.J)
        )

    def to_string(self) -> str:
        """Convert to string like 'M L T^-2'."""
        parts: list[str] = []
        mapping = {
            "M": self.M,
            "L": self.L,
            "T": self.T,
            "I": self.I,
            "Theta": self.Theta,
            "N": self.N,
            "J": self.J,
        }
        for name, exp in mapping.items():
            if exp == 0:
                continue
            if exp == 1:
                parts.append(name)
            else:
                parts.append(f"{name}^{exp}")
        return " ".join(parts) if parts else "dimensionless"

    @classmethod
    def from_string(cls, s: str) -> Dimension:
        """Parse a dimension string like 'M L T^-2' into a Dimension object.

        Supported formats:
        - 'M L T^-2'
        - 'M L T^{-2}'
        - 'dimensionless'
        - '' (empty = dimensionless)
        """
        s = s.strip()
        if not s or s.lower() == "dimensionless":
            return cls()

        kwargs: dict[str, int] = {}
        # Match tokens like 'Theta', 'M', 'L', 'T^-2', 'T^{-2}', 'Theta^2'
        # Theta must precede T so T doesn't greedily match the first letter of Theta
        pattern = r"(Theta|M|L|T|I|N|J)(?:\^[\{]?(-?\d+)[\}]?)?"
        for match in re.finditer(pattern, s):
            dim_name = match.group(1)
            exp = int(match.group(2)) if match.group(2) else 1
            kwargs[dim_name] = kwargs.get(dim_name, 0) + exp

        return cls(**kwargs)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"Dimension({self.to_string()})"


# ---------------------------------------------------------------------------
# PhysicsQuantity
# ---------------------------------------------------------------------------

class PhysicsQuantity(BaseModel):
    """A physical quantity definition or extracted value."""

    name: str
    symbol: str
    value: float | None = None
    unit: str | None = None
    dimension: str = ""
    si_unit: str = ""
    is_target: bool = False
    is_given: bool = False
    aliases: list[str] = Field(default_factory=list)

    @field_validator("symbol")
    @classmethod
    def symbol_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Symbol must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Equation
# ---------------------------------------------------------------------------

class Equation(BaseModel):
    """A physics equation in the knowledge base."""

    id: str
    name: str
    expression: str
    variables: list[str]
    variable_quantities: dict[str, str]
    domain: str
    topic: str
    description: str = ""
    conditions: list[str] = Field(default_factory=list)
    dimension_lhs: str = ""
    dimension_rhs: str = ""
    related_equations: list[str] = Field(default_factory=list)
    source: str = "textbook"

    @field_validator("expression")
    @classmethod
    def expression_has_equals(cls, v: str) -> str:
        if "=" not in v:
            raise ValueError(f"Equation expression must contain '=': {v}")
        return v.strip()


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class Problem(BaseModel):
    """A physics word problem for evaluation."""

    id: str
    problem_text: str
    topic: str
    difficulty: Difficulty
    source: ProblemSource
    given_quantities: list[PhysicsQuantity] = Field(default_factory=list)
    target_quantity: PhysicsQuantity | None = None
    required_equations: list[str] = Field(default_factory=list)
    answer_value: float
    answer_unit: str
    answer_tolerance: float = 0.01
    reasoning_steps: list[str] = Field(default_factory=list)
    distractors: list[str] = Field(default_factory=list)

    @field_validator("problem_text")
    @classmethod
    def problem_text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Problem text must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# LLM Parsed Output
# ---------------------------------------------------------------------------

class ParsedQuantity(BaseModel):
    """A quantity as parsed from LLM output."""

    name: str
    symbol: str
    value: float | None = None
    unit: str | None = None
    role: QuantityRole = QuantityRole.GIVEN


class ParsedEquation(BaseModel):
    """An equation as proposed by the LLM."""

    equation_id: str | None = None
    expression: str
    justification: str = ""


class LLMParsedOutput(BaseModel):
    """Structured output expected from the LLM."""

    problem_understanding: str = ""
    quantities: list[ParsedQuantity] = Field(default_factory=list)
    equations: list[ParsedEquation] = Field(default_factory=list)
    target_variable: str = ""
    solution_steps: list[str] = Field(default_factory=list)
    proposed_answer: float | None = None
    proposed_unit: str | None = None

    @model_validator(mode="after")
    def check_has_content(self) -> LLMParsedOutput:
        if not self.quantities and not self.equations:
            raise ValueError(
                "LLM output must contain at least quantities or equations"
            )
        return self


# ---------------------------------------------------------------------------
# Solver Result
# ---------------------------------------------------------------------------

class SolveResult(BaseModel):
    """Result of symbolic solving."""

    target_variable: str
    solutions: list[float | str] = Field(default_factory=list)
    is_numeric: bool = False
    num_solutions: int = 0
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_num_solutions(self) -> SolveResult:
        object.__setattr__(self, "num_solutions", len(self.solutions))
        return self


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class VerificationError(BaseModel):
    """A single verification error."""

    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    suggestion: str | None = None


class VerificationResult(BaseModel):
    """Result of the verification pipeline."""

    is_valid: bool
    errors: list[VerificationError] = Field(default_factory=list)
    checks_performed: list[str] = Field(default_factory=list)
    checks_passed: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Tool Call Record
# ---------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    """Record of a tool call for logging and analysis."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------

class SolveStep(BaseModel):
    """A single step in the solution process."""

    step_number: int
    action: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    tool_name: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: float = 0.0


class Solution(BaseModel):
    """Complete solution produced by the pipeline."""

    problem_id: str = Field(default_factory=lambda: str(uuid4()))
    answer_value: float | None = None
    answer_unit: str | None = None
    equations_used: list[str] = Field(default_factory=list)
    quantities_extracted: list[PhysicsQuantity] = Field(default_factory=list)
    solve_steps: list[SolveStep] = Field(default_factory=list)
    verification_result: VerificationResult | None = None
    num_attempts: int = 1
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    is_verified: bool = False
    error_message: str | None = None


# ---------------------------------------------------------------------------
# LLM Response
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    """Standard LLM response wrapper."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    model: str = ""
    finish_reason: str = ""
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Unit Conversion Result
# ---------------------------------------------------------------------------

class UnitConversionResult(BaseModel):
    """Result of a unit conversion."""

    value: float
    from_unit: str
    to_unit: str
    from_value: float
    to_value: float


# ---------------------------------------------------------------------------
# Dimension Check Result
# ---------------------------------------------------------------------------

class DimensionCheckResult(BaseModel):
    """Result of a dimensional consistency check."""

    is_consistent: bool
    lhs_dimension: str = ""
    rhs_dimension: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run."""

    experiment_name: str
    model_name: str
    system_config: dict[str, Any] = Field(default_factory=dict)
    dataset_split: str = "test"
    timestamp: datetime = Field(default_factory=datetime.now)
    random_seed: int = 42
    max_retries: int = 3


class ExperimentResult(BaseModel):
    """Results of running an experiment."""

    config: ExperimentConfig
    metrics: dict[str, float] = Field(default_factory=dict)
    per_problem_results: list[Solution] = Field(default_factory=list)
    total_problems: int = 0
    total_correct: int = 0
    total_duration_s: float = 0.0
    total_tokens_used: int = 0
    total_llm_calls: int = 0
