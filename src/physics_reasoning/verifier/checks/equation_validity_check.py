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

        for eq in parsed_output.equations:
            # 1. Check syntax
            try:
                parse_equation_string(eq.expression)
            except Exception as e:
                errors.append(
                    VerificationError(
                        error_type=ErrorType.SYNTAX_ERROR,
                        severity=ErrorSeverity.FATAL,
                        message=f"Equation '{eq.expression}' has invalid syntax: {e}",
                        context={"expression": eq.expression},
                        suggestion="Ensure equations use standard format like 'y = m * x + b'.",
                    )
                )

        return errors
