"""Symbolic equation solver using SymPy."""

from __future__ import annotations

import math
from typing import Any

import sympy
from sympy import Eq, Symbol, solve, symbols

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
        """Solve a single equation for the target variable with given known values.

        Example:
            solve_single("F = m * a", {"F": 10.0, "m": 2.0}, "a")
            -> SolveResult(target_variable="a", solutions=[5.0], is_numeric=True)

        Args:
            equation_str: String equation containing '=' (e.g. 'F = m * a').
            known_values: Mapping of variable names to known numerical values.
            target_variable: Name of variable to solve for (e.g. 'a').

        Returns:
            SolveResult containing solutions list and metadata.
        """
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
        2. Create substitutions for known variables.
        3. Substitute known values into equations.
        4. Solve the system for the target variable (and intermediate variables).
        5. Extract real, numeric solutions for target_variable.
        6. Return formatted SolveResult.

        Args:
            equations: List of equation strings.
            known_values: Dict mapping variable symbols to numeric values.
            target_variable: Target variable name.

        Returns:
            SolveResult
        """
        if not equations:
            return SolveResult(
                target_variable=target_variable,
                solutions=[],
                is_numeric=False,
                warnings=["No equations provided to solver"],
            )

        parsed_eqs: list[Eq] = []
        all_symbols: set[Symbol] = set()

        for eq_str in equations:
            try:
                eq = parse_equation_string(eq_str)
                parsed_eqs.append(eq)
                all_symbols.update(extract_symbols(eq))
            except Exception as e:
                return SolveResult(
                    target_variable=target_variable,
                    solutions=[],
                    is_numeric=False,
                    warnings=[f"Failed to parse equation '{eq_str}': {e}"],
                )

        target_sym = Symbol(target_variable)

        # Substitute known values
        subs_dict: dict[Symbol, Any] = {}
        for k, v in known_values.items():
            sym = Symbol(k)
            subs_dict[sym] = v

        subbed_eqs: list[Eq] = []
        for eq in parsed_eqs:
            subbed_lhs = eq.lhs.subs(subs_dict)
            subbed_rhs = eq.rhs.subs(subs_dict)
            subbed_eqs.append(Eq(subbed_lhs, subbed_rhs))

        # Solve system
        try:
            # We want to solve for target_sym, plus any other remaining unknown symbols
            remaining_symbols = set()
            for eq in subbed_eqs:
                remaining_symbols.update(eq.free_symbols)

            if target_sym not in remaining_symbols and target_sym in subs_dict:
                # Target was already in known values
                val = float(subs_dict[target_sym])
                return SolveResult(
                    target_variable=target_variable,
                    solutions=[val],
                    is_numeric=True,
                )

            raw_solutions = solve(subbed_eqs, list(remaining_symbols), dict=True)
        except Exception as e:
            return SolveResult(
                target_variable=target_variable,
                solutions=[],
                is_numeric=False,
                warnings=[f"SymPy solve failed: {e}"],
            )

        if not raw_solutions:
            # Fallback: try solving single equations directly if multiple equations failed together
            if len(subbed_eqs) == 1:
                try:
                    direct_solutions = solve(subbed_eqs[0], target_sym)
                    if direct_solutions:
                        raw_solutions = [{target_sym: sol} for sol in direct_solutions]
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

        for sol_dict in raw_solutions:
            if not isinstance(sol_dict, dict):
                # single var return format
                sol_val = sol_dict
            else:
                sol_val = sol_dict.get(target_sym)

            if sol_val is None:
                continue

            try:
                num_val = evaluate_numeric(sol_val)
                if math.isfinite(num_val):
                    # Dedup close solutions
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
        """Verify if given variable values satisfy an equation by computing residual.

        Args:
            equation_str: Equation like 'F = m * a'
            values: Mapping of all variables to values, e.g. {'F': 10, 'm': 2, 'a': 5}

        Returns:
            (is_satisfied, residual)
        """
        try:
            eq = parse_equation_string(equation_str)
        except Exception:
            return False, float("inf")

        subs_dict = {Symbol(k): v for k, v in values.items()}
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
