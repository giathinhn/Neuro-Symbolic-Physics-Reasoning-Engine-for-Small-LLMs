"""Verification pipeline orchestrator executing all enabled checks."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.core.enums import ErrorSeverity
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    SolveResult,
    VerificationError,
    VerificationResult,
)
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.solver.symbolic_solver import SymbolicSolver
from physics_reasoning.units.dimension_checker import DimensionChecker
from physics_reasoning.units.unit_engine import UnitEngine
from physics_reasoning.verifier.checks import BaseCheck
from physics_reasoning.verifier.checks.arithmetic_check import ArithmeticCheck
from physics_reasoning.verifier.checks.bounds_check import BoundsCheck
from physics_reasoning.verifier.checks.consistency_check import ConsistencyCheck
from physics_reasoning.verifier.checks.dimensional_check import DimensionalCheck
from physics_reasoning.verifier.checks.equation_validity_check import (
    EquationValidityCheck,
)
from physics_reasoning.verifier.checks.substitution_check import SubstitutionCheck
from physics_reasoning.verifier.checks.unit_check import UnitCheck


class VerificationPipeline:
    """Orchestrate all verification checks according to configuration."""

    def __init__(
        self,
        config: PipelineConfig,
        solver: SymbolicSolver | None = None,
        unit_engine: UnitEngine | None = None,
        dimension_checker: DimensionChecker | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ):
        self.config = config
        self.solver = solver or SymbolicSolver()
        self.unit_engine = unit_engine or UnitEngine()
        self.dim_checker = dimension_checker or DimensionChecker(self.unit_engine)
        self.kb = knowledge_base

        self.checks: list[BaseCheck] = []
        self._build_checks()

    def _build_checks(self) -> None:
        """Construct check list based on configuration toggles."""
        if self.config.enable_equation_validity_check:
            self.checks.append(EquationValidityCheck(self.kb))
        if self.config.enable_dimensional_check:
            self.checks.append(DimensionalCheck(self.dim_checker))
        if self.config.enable_unit_check:
            self.checks.append(UnitCheck(self.unit_engine))
        if self.config.enable_arithmetic_check:
            self.checks.append(ArithmeticCheck())
        if self.config.enable_bounds_check:
            self.checks.append(BoundsCheck())
        if self.config.enable_substitution_check:
            self.checks.append(SubstitutionCheck(self.solver))
        if self.config.enable_consistency_check:
            self.checks.append(ConsistencyCheck(self.solver))

    def verify(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str] | None = None,
        all_values: dict[str, float] | None = None,
    ) -> VerificationResult:
        """Execute all configured verification checks.

        Args:
            parsed_output: Parsed output from LLM.
            solve_result: Result from SymbolicSolver.
            equations_used: KnowledgeBase equations used.
            var_units: Mapping of variable names to unit strings.
            all_values: Mapping of all variables (given + computed target) to values.

        Returns:
            VerificationResult
        """
        var_units_dict = var_units or {}
        all_values_dict = all_values or {}

        # Populate missing var_units from parsed quantities
        for q in parsed_output.quantities:
            if q.symbol and q.unit and q.symbol not in var_units_dict:
                var_units_dict[q.symbol] = q.unit

        # Populate missing all_values from given quantities
        for q in parsed_output.quantities:
            if q.symbol and q.value is not None and q.symbol not in all_values_dict:
                all_values_dict[q.symbol] = q.value

        # If solver produced answer, add target to all_values
        if solve_result and solve_result.is_numeric and solve_result.solutions:
            first_sol = solve_result.solutions[0]
            if isinstance(first_sol, (int, float)):
                all_values_dict[solve_result.target_variable] = float(first_sol)

        all_errors: list[VerificationError] = []
        checks_performed: list[str] = []
        checks_passed: list[str] = []

        for check in self.checks:
            checks_performed.append(check.name)
            errors = check.run(
                parsed_output=parsed_output,
                solve_result=solve_result,
                equations_used=equations_used,
                var_units=var_units_dict,
                all_values=all_values_dict,
            )

            if errors:
                all_errors.extend(errors)
                # If any FATAL error encountered, stop executing further checks
                if any(e.severity == ErrorSeverity.FATAL for e in errors):
                    break
            else:
                checks_passed.append(check.name)

        has_blocking_errors = any(
            e.severity in (ErrorSeverity.ERROR, ErrorSeverity.FATAL) for e in all_errors
        )
        is_valid = not has_blocking_errors and (solve_result is not None and solve_result.is_numeric)

        # Confidence calculation
        if not checks_performed:
            confidence = 1.0
        else:
            confidence = len(checks_passed) / len(checks_performed)
            if has_blocking_errors:
                confidence = 0.0

        return VerificationResult(
            is_valid=is_valid,
            errors=all_errors,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            confidence=confidence,
            timestamp=datetime.now(),
        )
