"""System consistency verification check for multi-equation systems."""

from __future__ import annotations

from physics_reasoning.core.enums import ErrorSeverity, ErrorType
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    SolveResult,
    VerificationError,
)
from physics_reasoning.solver.symbolic_solver import SymbolicSolver
from physics_reasoning.verifier.checks import BaseCheck


class ConsistencyCheck(BaseCheck):
    """Verify that multiple equations in a system do not contradict each other."""

    def __init__(self, solver: SymbolicSolver | None = None):
        self.solver = solver or SymbolicSolver()

    @property
    def name(self) -> str:
        return "consistency"

    def run(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str],
        all_values: dict[str, float],
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        if len(parsed_output.equations) <= 1:
            return errors

        # If solver reported inconsistent system / no solution
        if solve_result and not solve_result.is_numeric and "no solution" in " ".join(solve_result.warnings).lower():
            errors.append(
                VerificationError(
                    error_type=ErrorType.INCONSISTENT_SYSTEM,
                    severity=ErrorSeverity.FATAL,
                    message="The system of equations is contradictory with no mathematical solution.",
                    context={"equations": [eq.expression for eq in parsed_output.equations]},
                    suggestion="Review problem constraints for conflicting equations.",
                )
            )

        return errors
