"""Arithmetic and numerical validity verification check."""

from __future__ import annotations

import math

from physics_reasoning.core.enums import ErrorSeverity, ErrorType
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    SolveResult,
    VerificationError,
)
from physics_reasoning.verifier.checks import BaseCheck


class ArithmeticCheck(BaseCheck):
    """Verify that solver produced real, finite, numerical answers."""

    @property
    def name(self) -> str:
        return "arithmetic"

    def run(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str],
        all_values: dict[str, float],
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        if solve_result is None:
            return errors

        if not solve_result.is_numeric or not solve_result.solutions:
            errors.append(
                VerificationError(
                    error_type=ErrorType.ARITHMETIC_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"Solver failed to produce a numeric answer for target '{solve_result.target_variable}'. Warnings: {solve_result.warnings}",
                    context={"warnings": solve_result.warnings},
                    suggestion="Verify that all required given values are provided and equations are solvable.",
                )
            )
            return errors

        # Check for non-finite values (NaN / Inf)
        for sol in solve_result.solutions:
            if isinstance(sol, (int, float)):
                if not math.isfinite(sol):
                    errors.append(
                        VerificationError(
                            error_type=ErrorType.ARITHMETIC_ERROR,
                            severity=ErrorSeverity.ERROR,
                            message=f"Computed answer is not a finite real number: {sol}",
                            context={"solution": sol},
                        )
                    )

        return errors
