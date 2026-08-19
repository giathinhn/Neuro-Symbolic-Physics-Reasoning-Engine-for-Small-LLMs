"""Back-substitution verification check."""

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


class SubstitutionCheck(BaseCheck):
    """Verify that calculated values satisfy the original equations upon back-substitution."""

    def __init__(self, solver: SymbolicSolver | None = None):
        self.solver = solver or SymbolicSolver()

    @property
    def name(self) -> str:
        return "substitution"

    def run(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str],
        all_values: dict[str, float],
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        if not all_values:
            return errors

        for eq in parsed_output.equations:
            is_sat, res = self.solver.verify_substitution(eq.expression, all_values)
            if not is_sat:
                errors.append(
                    VerificationError(
                        error_type=ErrorType.SUBSTITUTION_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message=f"Equation '{eq.expression}' is not satisfied by values {all_values} (residual = {res:.4e}).",
                        context={"equation": eq.expression, "residual": res, "values": all_values},
                        suggestion="Check the arithmetic or equation formulation.",
                    )
                )

        return errors
