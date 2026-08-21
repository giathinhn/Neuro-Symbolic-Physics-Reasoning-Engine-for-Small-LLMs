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

DEFAULT_PHYSICS_UNITS: dict[str, str] = {
    "I": "ampere",
    "I_1": "ampere",
    "I_2": "ampere",
    "i": "ampere",
    "current": "ampere",
    "U": "volt",
    "U_1": "volt",
    "U_2": "volt",
    "V": "volt",
    "voltage": "volt",
    "R": "ohm",
    "R_1": "ohm",
    "R_2": "ohm",
    "R_eq": "ohm",
    "R_td": "ohm",
    "resistance": "ohm",
    "F": "newton",
    "F_A": "newton",
    "F_g": "newton",
    "P_power": "watt",
    "P": "watt",
    "p": "pascal",
    "P_press": "pascal",
    "A": "meter ** 2",
    "A_work": "joule",
    "W": "joule",
    "W_work": "joule",
    "Q": "joule",
    "m": "kilogram",
    "m_1": "kilogram",
    "m_2": "kilogram",
    "m_hot": "kilogram",
    "m_cold": "kilogram",
    "m1": "kilogram",
    "m2": "kilogram",
    "v": "meter / second",
    "v_i": "meter / second",
    "v_f": "meter / second",
    "v_avg": "meter / second",
    "u": "meter / second",
    "s": "meter",
    "d": "meter",
    "h": "meter",
    "l": "meter",
    "x": "meter",
    "y": "meter",
    "S": "meter ** 2",
    "area": "meter ** 2",
    "V_vol": "meter ** 3",
    "t": "second",
    "time": "second",
    "g": "meter / second ** 2",
    "a": "meter / second ** 2",
    "c": "joule / (kilogram * kelvin)",
    "c_water": "joule / (kilogram * kelvin)",
    "c_1": "joule / (kilogram * kelvin)",
    "c_2": "joule / (kilogram * kelvin)",
    "c1": "joule / (kilogram * kelvin)",
    "c2": "joule / (kilogram * kelvin)",
    "T": "kelvin",
    "t_cb": "kelvin",
    "T_final": "kelvin",
    "T_hot": "kelvin",
    "T_cold": "kelvin",
    "t_1": "kelvin",
    "t_2": "kelvin",
    "t1": "kelvin",
    "t2": "kelvin",
    "rho": "kilogram / meter ** 3",
    "rho_water": "kilogram / meter ** 3",
    "rho_res": "ohm * meter",
    "d_spec": "newton / meter ** 3",
    "P_weight": "newton",
    "P_avg": "watt",
    "q_fuel": "joule / kilogram",
}


def _infer_unit_for_symbol(sym: str) -> str | None:
    """Infer physical unit from symbol naming conventions."""
    s = sym.lower()
    if s.startswith("p_weight") or s in ("p_weight", "p_trong_luong", "trong_luong"):
        return "newton"
    if s.startswith("f_") or s.endswith("_force") or "force" in s or s in ("f", "f_net", "f_a", "f_acs", "f_archimedes", "f_g", "f_drag", "f_pull", "f_push", "f_aerodynamic_force"):
        return "newton"
    if s.startswith("a_") or s.startswith("s_") or "area" in s or s in ("s", "a", "s_total", "a_total", "s_1", "s_2", "a_1", "a_2"):
        return "meter ** 2"
    if s.startswith("r_") or (s.startswith("r") and s[1:].isdigit()) or s in ("r", "r_eq", "r_td", "r_total"):
        return "ohm"
    if s.startswith("i_") or (s.startswith("i") and s[1:].isdigit()) or s in ("i", "i_total"):
        return "ampere"
    if s.startswith("u_") or (s.startswith("u") and s[1:].isdigit()) or s in ("u", "voltage"):
        return "volt"
    if s.startswith("p_press") or "press" in s or sym == "p" or s.startswith("p_ap"):
        return "pascal"
    if s.startswith("p_power") or "power" in s or sym.startswith("P_") or sym == "P" or "cong_suat" in s or s == "p_avg":
        return "watt"
    if s.startswith("q_fuel") or "nhien_lieu" in s:
        return "joule / kilogram"
    if s.startswith("rho_res") or "dien_tro_suat" in s:
        return "ohm * meter"
    if s.startswith("w_") or s.startswith("a_work") or "work" in s or s in ("a", "w", "q"):
        return "joule"
    if s.startswith("v_") or s in ("v", "u", "v_i", "v_f", "v_avg", "speed", "velocity"):
        return "meter / second"
    if s.startswith("m_") or (s.startswith("m") and s[1:].isdigit()) or s in ("m", "mass"):
        return "kilogram"
    if s.startswith("t_") and any(x in s for x in ["cb", "eq", "final", "hot", "cold", "temp", "celsius"]):
        return "kelvin"
    if s.startswith("t_") or s in ("t", "time", "delta_t"):
        return "second"
    if s.startswith("c_") or (s.startswith("c") and s[1:].isdigit()) or s in ("c", "specific_heat"):
        return "joule / (kilogram * kelvin)"
    if s.startswith("d_spec") or s.startswith("d_water") or s in ("d_spec", "d_water", "specific_weight", "trong_luong_rieng"):
        return "newton / meter ** 3"
    if s.startswith("rho") or "density" in s:
        return "kilogram / meter ** 3"
    if s in ("h", "d", "s", "x", "y", "l", "depth", "height", "distance", "path", "length", "width", "radius"):
        return "meter"
    if s in ("g", "a", "acceleration", "gravity"):
        return "meter / second ** 2"
    if s in ("v_vol", "v_total", "volume", "v"):
        return "meter ** 3"
    return None


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

        # Merge variable units with default units and physical constant units
        merged_units = dict(DEFAULT_PHYSICS_UNITS)
        merged_units.update(PHYSICAL_CONSTANT_UNITS)
        merged_units.update(variable_units)

        for sym in free_syms:
            unit_str = merged_units.get(sym)
            if not unit_str:
                unit_str = _infer_unit_for_symbol(sym)

            if unit_str:
                try:
                    norm_unit = self.unit_engine._normalize_unit_string(unit_str)
                    eval_context[sym] = self.ureg.Quantity(1.0, norm_unit)
                except Exception:
                    eval_context[sym] = 1.0
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

        # If rhs is 0 (e.g. Q_in - Q_out = 0), matching is satisfied if lhs is dimensionally valid
        if rhs_str in ("0", "0.0") and lhs_dim != Dimension():
            return DimensionCheckResult(
                is_consistent=True,
                lhs_dimension=lhs_dim.to_string(),
                rhs_dimension="0",
                message=f"Zero-residual equation is dimensionally consistent: [{lhs_dim.to_string()}]",
            )

        if lhs_dim == rhs_dim:
            return DimensionCheckResult(
                is_consistent=True,
                lhs_dimension=lhs_dim.to_string(),
                rhs_dimension=rhs_dim.to_string(),
                message=f"Equation is dimensionally consistent: [{lhs_dim.to_string()}] == [{rhs_dim.to_string()}]",
            )
        else:
            return DimensionCheckResult(
                is_consistent=False,
                lhs_dimension=lhs_dim.to_string(),
                rhs_dimension=rhs_dim.to_string(),
                message=(
                    f"Dimensional mismatch in equation '{equation_str}': "
                    f"LHS has dimension [{lhs_dim.to_string()}], but RHS has dimension [{rhs_dim.to_string()}]."
                ),
            )
