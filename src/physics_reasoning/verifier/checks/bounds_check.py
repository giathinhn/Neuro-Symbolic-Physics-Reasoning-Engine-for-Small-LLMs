"""Physical domain constraints and bounds verification check."""

from __future__ import annotations

from physics_reasoning.core.enums import ErrorSeverity, ErrorType
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    SolveResult,
    VerificationError,
)
from physics_reasoning.solver.numerical import is_physically_reasonable
from physics_reasoning.verifier.checks import BaseCheck


class BoundsCheck(BaseCheck):
    """Verify that calculated and given values satisfy physical constraints."""

    @property
    def name(self) -> str:
        return "bounds"

    def run(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str],
        all_values: dict[str, float],
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        # Check all values in all_values against physical bounds
        for var_name, val in all_values.items():
            # Find matching quantity name if possible
            matching_q = next((q for q in parsed_output.quantities if q.symbol == var_name), None)
            q_name = matching_q.name if matching_q else var_name

            ok, reason = is_physically_reasonable(val, q_name)
            if not ok and reason:
                errors.append(
                    VerificationError(
                        error_type=ErrorType.IMPOSSIBLE_VALUE,
                        severity=ErrorSeverity.ERROR,
                        message=f"Value for '{var_name}' ({q_name} = {val}) violates physical bounds: {reason}",
                        context={"variable": var_name, "value": val, "reason": reason},
                        suggestion="Check algebraic signs or whether root selection was correct.",
                    )
                )

        return errors
