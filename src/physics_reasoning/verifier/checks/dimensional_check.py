"""Dimensional consistency verification check."""

from __future__ import annotations

from physics_reasoning.core.enums import ErrorSeverity, ErrorType
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    SolveResult,
    VerificationError,
)
from physics_reasoning.units.dimension_checker import DimensionChecker
from physics_reasoning.verifier.checks import BaseCheck


class DimensionalCheck(BaseCheck):
    """Verify that all proposed equations are dimensionally consistent."""

    def __init__(self, dimension_checker: DimensionChecker | None = None):
        self.dim_checker = dimension_checker or DimensionChecker()

    @property
    def name(self) -> str:
        return "dimensional"

    def run(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str],
        all_values: dict[str, float],
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        # Check each equation proposed by LLM
        for eq in parsed_output.equations:
            res = self.dim_checker.check_equation(eq.expression, var_units)
            if not res.is_consistent:
                errors.append(
                    VerificationError(
                        error_type=ErrorType.DIMENSION_MISMATCH,
                        severity=ErrorSeverity.ERROR,
                        message=res.message,
                        context={
                            "equation": eq.expression,
                            "lhs_dimension": res.lhs_dimension,
                            "rhs_dimension": res.rhs_dimension,
                        },
                        suggestion=f"Check equation '{eq.expression}' for missing powers or incorrect formula.",
                    )
                )

        return errors
