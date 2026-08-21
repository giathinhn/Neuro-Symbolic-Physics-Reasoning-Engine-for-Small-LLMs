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

        # Thermodynamic Equilibrium Check (2nd Law of Thermodynamics)
        # Without external heat source, final equilibrium temperature MUST lie between min and max initial temperatures
        eq_temp_symbols = {"t_cb", "tf", "t_f", "t_final", "t_final", "t_equilibrium", "t_cb", "temperature_equilibrium"}
        target_sym = parsed_output.target_variable
        
        eq_temp_val: float | None = None
        for s in eq_temp_symbols | {target_sym}:
            if s in all_values:
                # Check if this variable represents temperature
                unit = var_units.get(s, "").lower()
                if unit in ("celsius", "kelvin", "degc", "°c", "độ c", "k") or s in eq_temp_symbols or "temp" in s.lower():
                    eq_temp_val = all_values[s]
                    break

        if eq_temp_val is not None:
            init_temps: list[float] = []
            for k, v in all_values.items():
                if k in ("t_1", "t_2", "t1", "t2", "t_hot", "t_cold", "t_hot_initial", "t_cold_initial") or (
                    k.startswith("t") and k not in eq_temp_symbols and k != target_sym and var_units.get(k, "").lower() in ("celsius", "kelvin", "degc", "°c", "k")
                ):
                    init_temps.append(v)

            if len(init_temps) >= 2:
                t_min = min(init_temps)
                t_max = max(init_temps)
                # Allow tiny numerical tolerance of 1e-3
                if eq_temp_val < t_min - 1e-3 or eq_temp_val > t_max + 1e-3:
                    errors.append(
                        VerificationError(
                            error_type=ErrorType.IMPOSSIBLE_VALUE,
                            severity=ErrorSeverity.ERROR,
                            message=(
                                f"Equilibrium temperature ({eq_temp_val:.2f}) violates thermodynamic bounds: "
                                f"it must lie strictly between the initial temperatures [{t_min:.2f}, {t_max:.2f}]."
                            ),
                            context={"equilibrium_temperature": eq_temp_val, "t_min": t_min, "t_max": t_max},
                            suggestion=(
                                "Check equation indexing: ensure each mass is multiplied by its OWN initial temperature difference, "
                                "e.g. m_1 * (t_cb - t_1) + m_2 * (t_cb - t_2) = 0."
                            ),
                        )
                    )

        # Parallel Resistor Bound Check: R_eq < min(R_1, R_2, ...)
        r_eq_val = all_values.get("R_eq") or all_values.get("r_eq") or (all_values.get(target_sym) if var_units.get(target_sym, "").lower() == "ohm" else None)
        if r_eq_val is not None:
            branch_rs = [v for k, v in all_values.items() if k in ("R_1", "R_2", "r1", "r2", "r_1", "r_2") and v > 0]
            is_parallel = any("parallel" in eq.topic.lower() or "parallel" in eq.name.lower() or "parallel" in eq.id.lower() for eq in equations_used)
            if is_parallel and len(branch_rs) >= 2:
                min_r = min(branch_rs)
                if r_eq_val >= min_r + 1e-3:
                    errors.append(
                        VerificationError(
                            error_type=ErrorType.IMPOSSIBLE_VALUE,
                            severity=ErrorSeverity.ERROR,
                            message=f"Equivalent resistance for parallel circuit ({r_eq_val:.2f} ohm) must be strictly less than individual branch resistance ({min_r:.2f} ohm).",
                            context={"r_eq": r_eq_val, "min_branch_r": min_r},
                            suggestion="Use parallel resistance formula: R_eq = (R_1 * R_2) / (R_1 + R_2).",
                        )
                    )

        return errors
