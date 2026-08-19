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
        """Solve a system of equations for the target variable given known values.

        Algorithm:
        1. Parse all equation strings into SymPy Eq objects.
        2. Substitute known values into equations.
        3. Solve the system for all unknown symbols (eliminating intermediate variables).
        4. Extract real, numeric solutions for target_variable.
        5. Return formatted SolveResult.
        """
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
            except Exception as e:
                return SolveResult(
                    target_variable=target_variable,
                    solutions=[],
                    is_numeric=False,
                    warnings=[f"Failed to parse equation '{eq_str}': {e}"],
                )

        target_sym = Symbol(target_variable)

        # Build substitution dictionary
        subs_dict: dict[Symbol, Any] = {}
        for k, v in known_values.items():
            sym = Symbol(k)
            if isinstance(v, (int, float)):
                subs_dict[sym] = sympy.Float(v)
            else:
                subs_dict[sym] = v

        subbed_eqs: list[Eq] = []
        for eq in parsed_eqs:
            subbed_lhs = eq.lhs.subs(subs_dict)
            subbed_rhs = eq.rhs.subs(subs_dict)
            subbed_eqs.append(Eq(subbed_lhs, subbed_rhs))

        # Check if target was already directly known
        if str(target_sym) in known_values:
            val = float(known_values[str(target_sym)])
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

        # Solve system for all remaining unknowns simultaneously
        raw_solutions: list[Any] = []
        try:
            sol = solve(subbed_eqs, list(remaining_symbols), dict=True)
            if isinstance(sol, list):
                raw_solutions.extend(sol)
            elif isinstance(sol, dict):
                raw_solutions.append(sol)
        except Exception:
            pass

        # Fallback if simultaneous solve with full symbol list didn't yield dict solutions
        if not raw_solutions:
            try:
                # Try solve without restricting symbols
                sol = solve(subbed_eqs, dict=True)
                if isinstance(sol, list):
                    raw_solutions.extend(sol)
            except Exception:
                pass

        # Second fallback: solve single equation for target_sym directly
        if not raw_solutions and len(subbed_eqs) == 1:
            try:
                sol = solve(subbed_eqs[0], target_sym)
                if isinstance(sol, list):
                    for s in sol:
                        raw_solutions.append({target_sym: s})
                elif sol is not None:
                    raw_solutions.append({target_sym: sol})
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

        for sol_item in raw_solutions:
            sol_val = None
            if isinstance(sol_item, dict):
                for k, v in sol_item.items():
                    if str(k) == target_variable or k == target_sym:
                        sol_val = v
                        break
            else:
                sol_val = sol_item

            if sol_val is None:
                continue

            try:
                num_val = evaluate_numeric(sol_val)
                if math.isfinite(num_val):
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

        subs_dict = {Symbol(k): nsimplify(v) for k, v in values.items()}
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
