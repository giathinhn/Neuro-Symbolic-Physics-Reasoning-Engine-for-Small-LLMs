"""Dimensional analysis and equation consistency checker using Pint and SymPy."""

from __future__ import annotations

import re
from typing import Any

from pint import DimensionalityError, Quantity

from physics_reasoning.core.exceptions import ExpressionParseError
from physics_reasoning.core.models import Dimension, DimensionCheckResult
from physics_reasoning.physics.constants import PHYSICAL_CONSTANT_UNITS
from physics_reasoning.solver.expression_parser import (
    extract_symbols,
    parse_equation_string,
    parse_expression,
)
from physics_reasoning.units.unit_engine import UnitEngine


class DimensionChecker:
    """Check dimensional consistency of physics equations and expressions."""

    def __init__(self, unit_engine: UnitEngine | None = None):
        self.unit_engine = unit_engine or UnitEngine()
        self.ureg = self.unit_engine.ureg

    def get_expression_dimension_object(
        self, expr_str: str, variable_units: dict[str, str]
    ) -> Dimension:
        """Compute the Dimension of an algebraic expression given units for its variables."""
        sympy_expr = parse_expression(expr_str)
        free_syms = [str(s) for s in sympy_expr.free_symbols]

        # Build context dict with Pint Quantities
        eval_context: dict[str, Any] = {
            "sqrt": lambda x: x ** 0.5,
            "sin": lambda x: 1.0,
            "cos": lambda x: 1.0,
            "tan": lambda x: 1.0,
            "abs": abs,
            "Abs": abs,
            "pi": 1.0,
        }

        # Merge variable units with known physical constant units
        merged_units = dict(PHYSICAL_CONSTANT_UNITS)
        merged_units.update(variable_units)

        for sym in free_syms:
            unit_str = merged_units.get(sym)
            if unit_str:
                norm_unit = self.unit_engine._normalize_unit_string(unit_str)
                eval_context[sym] = self.ureg.Quantity(1.0, norm_unit)
            else:
                eval_context[sym] = 1.0

        safe_eval_str = str(sympy_expr)
        try:
            val = eval(safe_eval_str, {"__builtins__": {}}, eval_context)
            if isinstance(val, Quantity):
                dim_dict = val.dimensionality
                return Dimension(
                    M=int(dim_dict.get("[mass]", 0)),
                    L=int(dim_dict.get("[length]", 0)),
                    T=int(dim_dict.get("[time]", 0)),
                    I=int(dim_dict.get("[current]", 0)),
                    Theta=int(dim_dict.get("[temperature]", 0)),
                    N=int(dim_dict.get("[substance]", 0)),
                    J=int(dim_dict.get("[luminosity]", 0)),
                )
            else:
                return Dimension()
        except Exception:
            return Dimension()

    def get_expression_dimension(
        self, expr_str: str, variable_units: dict[str, str]
    ) -> str:
        """Compute the dimension string (e.g. 'M L T^-2') for an expression."""
        return self.get_expression_dimension_object(expr_str, variable_units).to_string()

    def check_equation(
        self,
        equation_str: str,
        variable_units: dict[str, str],
    ) -> DimensionCheckResult:
        """Check if an equation is dimensionally consistent."""
        try:
            eq = parse_equation_string(equation_str)
        except Exception as e:
            return DimensionCheckResult(
                is_consistent=False,
                lhs_dimension="",
                rhs_dimension="",
                message=f"Failed to parse equation: {e}",
            )

        lhs_str = str(eq.lhs)
        rhs_str = str(eq.rhs)

        lhs_dim = self.get_expression_dimension_object(lhs_str, variable_units)
        rhs_dim = self.get_expression_dimension_object(rhs_str, variable_units)

        is_consistent = (lhs_dim == rhs_dim)

        if is_consistent:
            msg = f"Equation is dimensionally consistent: [{lhs_dim.to_string()}] == [{rhs_dim.to_string()}]"
        else:
            msg = (
                f"Dimensional mismatch in equation '{equation_str}': "
                f"LHS has dimension [{lhs_dim.to_string()}], "
                f"but RHS has dimension [{rhs_dim.to_string()}]."
            )

        return DimensionCheckResult(
            is_consistent=is_consistent,
            lhs_dimension=lhs_dim.to_string(),
            rhs_dimension=rhs_dim.to_string(),
            message=msg,
        )
