"""Verification checks subpackage."""

from __future__ import annotations

from abc import ABC, abstractmethod

from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    SolveResult,
    VerificationError,
)


class BaseCheck(ABC):
    """Abstract base class for all individual verification checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier for this check."""
        ...

    @abstractmethod
    def run(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str],
        all_values: dict[str, float],
    ) -> list[VerificationError]:
        """Execute verification check.

        Args:
            parsed_output: Structured LLM output.
            solve_result: Output from SymbolicSolver.
            equations_used: Matched knowledge base equations.
            var_units: Mapping of variable names to units.
            all_values: Mapping of all variables (given + computed target) to numeric values.

        Returns:
            List of VerificationError objects (empty if check passed).
        """
        ...
