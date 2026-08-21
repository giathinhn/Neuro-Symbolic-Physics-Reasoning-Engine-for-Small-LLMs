"""Symbolic equation solver using SymPy."""

from __future__ import annotations

import math
from typing import Any

import sympy
from sympy import Eq, Symbol, nsimplify, solve, symbols

from physics_reasoning.core.exceptions import (
    ExpressionParseError,
    NoSolutionError,
    SolverError,
)
from physics_reasoning.core.models import SolveResult
from physics_reasoning.solver.expression_parser import (
    extract_symbols,
    parse_equation_string,
    parse_expression,
)
from physics_reasoning.solver.numerical import evaluate_numeric
PHYSICS_SYNONYM_GROUPS: list[set[str]] = [
    {"h", "s", "y", "x", "l", "depth", "height", "distance", "path", "quang_duong"},
    {"S", "S_total", "A_total", "area", "total_area", "dien_tich"},
    {"S_1", "A_1", "S1", "A1", "single_area", "dien_tich_1"},
    {"S_2", "A_2", "S2", "A2", "dien_tich_2"},
    {"F", "F_net", "force", "F_pull", "F_push", "F_g", "P_weight", "weight", "trong_luong"},
    {"F_A", "F_acs", "F_archimedes", "buoyant_force", "F_aerodynamic_force"},
    {"d", "d_spec", "d_water", "d_liquid", "specific_weight", "trong_luong_rieng"},
    {"P", "P_power", "power", "cong_suat", "P_crane", "P_engine"},
    {"p", "P_press", "pressure", "ap_suat"},
    {"A", "W_work", "work", "cong"},
    {"t", "time", "delta_t"},
    {"v", "v_i", "v_avg", "u", "speed", "velocity"},
    {"t_cb", "T_final", "t_final", "T_cb", "t_eq", "T_eq", "T_equilibrium", "Tf", "tf", "T_f"},
    {"t_1", "T_1", "t1", "T1", "T_hot", "t_hot", "temperature_1"},
    {"t_2", "T_2", "t2", "T2", "T_cold", "t_cold", "temperature_2"},
    {"m", "mass"},
    {"m_1", "m1", "m_hot", "mass_1"},
    {"m_2", "m2", "m_cold", "mass_2"},
    {"c_water", "c_heat", "c_specific", "specific_heat", "nhiet_dung_rieng"},
    {"c_1", "c1"},
    {"c_2", "c2"},
    {"Q", "Q_toa", "Q_thu", "heat", "nhiet_luong"},
    {"R_eq", "R_td", "R_total", "R_equivalent"},
    {"R", "resistance", "r"},
    {"R_1", "R1", "resistance_1"},
    {"R_2", "R2", "resistance_2"},
    {"I", "current", "i"},
    {"I_1", "I1", "current_1"},
    {"I_2", "I2", "current_2"},
    {"U", "V", "voltage", "u"},
    {"U_1", "U1", "voltage_1"},
    {"U_2", "U2", "voltage_2"},
]


def _find_synonyms(sym_str: str) -> set[str]:
    """Find all synonymous symbol names for a given symbol."""
    res = {sym_str}
    for group in PHYSICS_SYNONYM_GROUPS:
        if sym_str in group:
            res.update(group)
    return res


class SymbolicSolver:
    """Symbolic and numerical equation solver based on SymPy."""

    def __init__(self, numerical_tolerance: float = 1e-6, timeout_seconds: float = 10.0):
        self.numerical_tolerance = numerical_tolerance
        self.timeout_seconds = timeout_seconds

    def solve_single(
        self,
        equation_str: str,
        known_values: dict[str, float],
        target_variable: str,
    ) -> SolveResult:
        """Solve a single equation for the target variable with given known values."""
        return self.solve_system(
            equations=[equation_str],
            known_values=known_values,
            target_variable=target_variable,
        )

    def solve_system(
        self,
        equations: list[str],
        known_values: dict[str, float],
        target_variable: str,
    ) -> SolveResult:
        """Solve a system of equations for the target variable given known values."""
        if not equations:
            return SolveResult(
                target_variable=target_variable,
                solutions=[],
                is_numeric=False,
                warnings=["No equations provided to solver"],
            )

        parsed_eqs: list[Eq] = []
        for eq_str in equations:
            try:
                eq = parse_equation_string(eq_str)
                parsed_eqs.append(eq)
            except Exception:
                pass

        if not parsed_eqs:
            return SolveResult(
                target_variable=target_variable,
                solutions=[],
                is_numeric=False,
                warnings=["No valid equations could be parsed"],
            )

        target_syms = {Symbol(s) for s in _find_synonyms(target_variable)}

        # Build substitution dictionary with synonym support
        subs_dict: dict[Symbol, Any] = {}
        # First add direct knowns
        for k, v in known_values.items():
            sym = Symbol(k)
            if isinstance(v, (int, float)):
                subs_dict[sym] = sympy.Float(v)
            else:
                subs_dict[sym] = v

        # For all free symbols across equations, if not in known_values, check synonyms
        all_eq_symbols = set()
        for eq in parsed_eqs:
            all_eq_symbols.update(eq.free_symbols)

        for sym in all_eq_symbols:
            sym_name = str(sym)
            if sym not in subs_dict and sym not in target_syms:
                synonyms = _find_synonyms(sym_name)
                for syn in synonyms:
                    if syn in known_values and Symbol(syn) not in all_eq_symbols:
                        v = known_values[syn]
                        subs_dict[sym] = sympy.Float(v) if isinstance(v, (int, float)) else v
                        break

        subbed_eqs: list[Eq] = []
        for eq in parsed_eqs:
            subbed_lhs = eq.lhs.subs(subs_dict)
            subbed_rhs = eq.rhs.subs(subs_dict)
            subbed_eqs.append(Eq(subbed_lhs, subbed_rhs))

        # Check if target was already directly known
        for ts in target_syms:
            if str(ts) in known_values:
                val = float(known_values[str(ts)])
                return SolveResult(
                    target_variable=target_variable,
                    solutions=[val],
                    is_numeric=True,
                )

        # Collect all remaining free symbols
        remaining_symbols: set[Symbol] = set()
        for eq in subbed_eqs:
            remaining_symbols.update(eq.free_symbols)

        if not remaining_symbols:
            return SolveResult(
                target_variable=target_variable,
                solutions=[],
                is_numeric=False,
                warnings=["No unknown variables remain to solve"],
            )

        # Solve system for all remaining unknowns
        raw_solutions: list[Any] = []
        try:
            sol = solve(subbed_eqs, dict=True)
            if isinstance(sol, list):
                raw_solutions.extend(sol)
            elif isinstance(sol, dict):
                raw_solutions.append(sol)
        except Exception:
            pass

        # Fallback with explicit symbol list if dict=True yielded nothing
        if not raw_solutions:
            try:
                sol = solve(subbed_eqs, list(remaining_symbols), dict=True)
                if isinstance(sol, list):
                    raw_solutions.extend(sol)
                elif isinstance(sol, dict):
                    raw_solutions.append(sol)
            except Exception:
                pass

        # Second fallback: solve single equation for target_sym directly (only when 1 equation given)
        if not raw_solutions and len(subbed_eqs) == 1:
            for ts in target_syms:
                for sub_eq in subbed_eqs:
                    if ts in sub_eq.free_symbols:
                        try:
                            sol = solve(sub_eq, ts)
                            if isinstance(sol, list):
                                for s in sol:
                                    raw_solutions.append({ts: s})
                            elif sol is not None:
                                raw_solutions.append({ts: sol})
                        except Exception:
                            pass

        if not raw_solutions:
            return SolveResult(
                target_variable=target_variable,
                solutions=[],
                is_numeric=False,
                warnings=["No solution found for equation system with provided values"],
            )

        # Process and filter solutions
        numeric_solutions: list[float] = []
        symbolic_solutions: list[str] = []
        warnings: list[str] = []

        target_names = {str(ts) for ts in target_syms}

        for sol_item in raw_solutions:
            sol_val = None
            if isinstance(sol_item, dict):
                for k, v in sol_item.items():
                    if str(k) in target_names or k in target_syms:
                        sol_val = v
                        break
            else:
                sol_val = sol_item

            if sol_val is None:
                continue

            try:
                num_val = evaluate_numeric(sol_val)
                if math.isfinite(num_val):
                    # Filter spurious 0.0 if all known inputs are positive and target is non-zero quantity
                    all_known_positive = all(isinstance(v, (int, float)) and v > 0 for v in known_values.values()) if known_values else False
                    if math.isclose(num_val, 0.0, abs_tol=1e-9) and all_known_positive and target_variable.lower() in ("p", "p_avg", "p_power", "a", "w", "w_work", "f", "r", "r_eq", "i", "u", "v", "d", "s"):
                        # Check if this 0.0 is from an underdetermined trivial kernel
                        if len(raw_solutions) > 1 or any(isinstance(s, dict) and any(not isinstance(val, (int, float, sympy.Number)) for val in s.values()) for s in raw_solutions):
                            continue
                    if not any(
                        math.isclose(num_val, existing, abs_tol=self.numerical_tolerance)
                        for existing in numeric_solutions
                    ):
                        numeric_solutions.append(num_val)
            except Exception:
                symbolic_solutions.append(str(sol_val))

        if numeric_solutions:
            return SolveResult(
                target_variable=target_variable,
                solutions=numeric_solutions,
                is_numeric=True,
                warnings=warnings,
            )
        elif symbolic_solutions:
            return SolveResult(
                target_variable=target_variable,
                solutions=symbolic_solutions,
                is_numeric=False,
                warnings=warnings + ["Solutions are symbolic (underdetermined system)"],
            )
        else:
            return SolveResult(
                target_variable=target_variable,
                solutions=[],
                is_numeric=False,
                warnings=["Could not extract real numeric solutions for target variable"],
            )

    def verify_substitution(
        self,
        equation_str: str,
        values: dict[str, float],
    ) -> tuple[bool, float]:
        """Verify if given variable values satisfy an equation by computing residual."""
        try:
            eq = parse_equation_string(equation_str)
        except Exception:
            return False, float("inf")

        subs_dict = {}
        for k, v in values.items():
            if isinstance(v, (int, float)):
                subs_dict[Symbol(k)] = nsimplify(v)
                for syn in _find_synonyms(k):
                    if Symbol(syn) not in subs_dict:
                        subs_dict[Symbol(syn)] = nsimplify(v)
            else:
                subs_dict[Symbol(k)] = v

        lhs_val = eq.lhs.subs(subs_dict)
        rhs_val = eq.rhs.subs(subs_dict)

        try:
            lhs_num = evaluate_numeric(lhs_val)
            rhs_num = evaluate_numeric(rhs_val)
            diff = abs(lhs_num - rhs_num)
            max_mag = max(abs(lhs_num), abs(rhs_num), 1.0)
            rel_diff = diff / max_mag

            is_sat = diff <= self.numerical_tolerance or rel_diff <= 0.01
            return is_sat, diff
        except Exception:
            return False, float("inf")
