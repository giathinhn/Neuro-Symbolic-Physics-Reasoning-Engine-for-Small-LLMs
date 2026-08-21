"""Equation validity and syntactic structure check."""

from __future__ import annotations

from physics_reasoning.core.enums import ErrorSeverity, ErrorType
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    SolveResult,
    VerificationError,
)
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.solver.expression_parser import parse_equation_string
from physics_reasoning.verifier.checks import BaseCheck


class EquationValidityCheck(BaseCheck):
    """Verify syntactic correctness and validity of proposed equations."""

    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self.kb = knowledge_base

    @property
    def name(self) -> str:
        return "equation_validity"

    def run(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str],
        all_values: dict[str, float],
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        if not parsed_output.equations:
            errors.append(
                VerificationError(
                    error_type=ErrorType.INVALID_EQUATION,
                    severity=ErrorSeverity.FATAL,
                    message="No equations were proposed by the model to solve the problem.",
                    suggestion="Identify relevant physics formulas relating known quantities to the target.",
                )
            )
            return errors

        valid_count = 0
        for eq in parsed_output.equations:
            # Check syntax
            try:
                parse_equation_string(eq.expression)
                valid_count += 1
            except Exception as e:
                # If expression looks like a unit constant assignment e.g. "rho = 1000 kg/m^3", treat as info/warning
                errors.append(
                    VerificationError(
                        error_type=ErrorType.SYNTAX_ERROR,
                        severity=ErrorSeverity.WARNING,
                        message=f"Equation '{eq.expression}' has non-standard syntax: {e}",
                        context={"expression": eq.expression},
                        suggestion="Ensure equations use standard algebraic format like 'y = m * x + b'.",
                    )
                )

        if valid_count == 0 and errors:
            # Escalate the first error to FATAL if no equation was valid
            errors[0].severity = ErrorSeverity.FATAL

        return errors
