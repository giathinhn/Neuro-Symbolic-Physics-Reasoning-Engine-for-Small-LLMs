"""Unit consistency verification check."""

from __future__ import annotations

from physics_reasoning.core.enums import ErrorSeverity, ErrorType, QuantityRole
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    SolveResult,
    VerificationError,
)
from physics_reasoning.units.unit_engine import UnitEngine
from physics_reasoning.verifier.checks import BaseCheck


class UnitCheck(BaseCheck):
    """Verify that quantity units are valid and dimensionally compatible."""

    def __init__(self, unit_engine: UnitEngine | None = None):
        self.unit_engine = unit_engine or UnitEngine()

    @property
    def name(self) -> str:
        return "unit"

    def run(
        self,
        parsed_output: LLMParsedOutput,
        solve_result: SolveResult | None,
        equations_used: list[Equation],
        var_units: dict[str, str],
        all_values: dict[str, float],
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        # Check that units of all quantities can be parsed by Pint
        for q in parsed_output.quantities:
            if q.unit:
                try:
                    self.unit_engine.parse_unit(q.unit)
                except Exception as e:
                    errors.append(
                        VerificationError(
                            error_type=ErrorType.UNIT_MISMATCH,
                            severity=ErrorSeverity.ERROR,
                            message=f"Invalid unit '{q.unit}' for quantity '{q.name}': {e}",
                            context={"quantity": q.name, "unit": q.unit},
                            suggestion="Use standard unit notation (e.g. 'm/s', 'kg', 'N', 'J').",
                        )
                    )

        # Check target unit compatibility with proposed answer unit if specified
        target_quantities = [q for q in parsed_output.quantities if q.role == QuantityRole.TARGET]
        if target_quantities and parsed_output.proposed_unit:
            target_unit = target_quantities[0].unit
            if target_unit and not self.unit_engine.are_compatible(target_unit, parsed_output.proposed_unit):
                errors.append(
                    VerificationError(
                        error_type=ErrorType.UNIT_MISMATCH,
                        severity=ErrorSeverity.ERROR,
                        message=(
                            f"Proposed answer unit '{parsed_output.proposed_unit}' is incompatible with "
                            f"target unit '{target_unit}'."
                        ),
                        context={"target_unit": target_unit, "proposed_unit": parsed_output.proposed_unit},
                    )
                )

        return errors
