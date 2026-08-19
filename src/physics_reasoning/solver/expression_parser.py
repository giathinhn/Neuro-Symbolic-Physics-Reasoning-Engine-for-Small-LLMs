"""Expression parser for converting string equations/math to SymPy objects safely."""

from __future__ import annotations

import re
from typing import Any

import sympy
from sympy import Eq, Expr, Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from physics_reasoning.core.exceptions import ExpressionParseError

# Standard safe transformations for physics expressions
SAFE_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Whitelist of allowed SymPy mathematical functions and constants
ALLOWED_NAMES: dict[str, Any] = {
    "pi": sympy.pi,
    "E": sympy.E,
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "abs": sympy.Abs,
    "Abs": sympy.Abs,
    "log": sympy.log,
    "ln": sympy.log,
    "exp": sympy.exp,
}

# Regex to detect potentially malicious code patterns
DISALLOWED_PATTERNS = [
    r"__\w+__",
    r"\bimport\b",
    r"\bexec\b",
    r"\beval\b",
    r"\bcompile\b",
    r"\bopen\b",
    r"\bos\b",
    r"\bsys\b",
    r"\bsubprocess\b",
]


def _validate_safety(expr_str: str) -> None:
    """Check for suspicious strings or python code injection attempts."""
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, expr_str, flags=re.IGNORECASE):
            raise ExpressionParseError(
                f"Expression contains disallowed pattern '{pattern}': {expr_str}",
                expression=expr_str,
            )


def parse_expression(
    expr_str: str, local_symbols: dict[str, Symbol] | None = None
) -> Expr:
    """Parse a string into a SymPy expression safely.

    Features:
    - Whitelisted names and symbols
    - Implicit multiplication (e.g., '2a' -> 2*a, 'm a' -> m*a)
    - Power syntax conversion ('^' -> '**')
    - Subscript variables ('v_f', 'F_net')
    - Anti-code-injection validation

    Args:
        expr_str: The mathematical expression string.
        local_symbols: Optional mapping of custom variable names to Symbols.

    Returns:
        SymPy Expr.

    Raises:
        ExpressionParseError: If parsing fails or input is disallowed.
    """
    if not expr_str or not expr_str.strip():
        raise ExpressionParseError("Cannot parse empty expression", expression=expr_str)

    clean_str = expr_str.strip()
    _validate_safety(clean_str)

    # Prepare local dict combining allowed names with provided symbols
    local_dict: dict[str, Any] = dict(ALLOWED_NAMES)
    if local_symbols:
        local_dict.update(local_symbols)

    try:
        parsed = parse_expr(
            clean_str,
            local_dict=local_dict,
            transformations=SAFE_TRANSFORMATIONS,
            evaluate=True,
        )
        return parsed
    except Exception as e:
        raise ExpressionParseError(
            f"Failed to parse expression '{expr_str}': {e}", expression=expr_str
        ) from e


def parse_equation_string(
    eq_str: str, local_symbols: dict[str, Symbol] | None = None
) -> Eq:
    """Parse an equation string containing '=' into a SymPy Eq(lhs, rhs).

    Example:
        'F = m * a' -> Eq(F, m*a)

    Args:
        eq_str: The equation string with '='.
        local_symbols: Optional mapping of custom variable names to Symbols.

    Returns:
        SymPy Eq object.

    Raises:
        ExpressionParseError: If '=' is missing or either side fails to parse.
    """
    if not eq_str or not eq_str.strip():
        raise ExpressionParseError("Cannot parse empty equation", expression=eq_str)

    clean_str = eq_str.strip()
    _validate_safety(clean_str)

    if "=" not in clean_str:
        raise ExpressionParseError(
            f"Equation string must contain '=': {eq_str}", expression=eq_str
        )

    parts = clean_str.split("=")
    if len(parts) != 2:
        raise ExpressionParseError(
            f"Equation must contain exactly one '=' sign: {eq_str}",
            expression=eq_str,
        )

    lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
    if not lhs_str or not rhs_str:
        raise ExpressionParseError(
            f"Both sides of equation must be non-empty: {eq_str}",
            expression=eq_str,
        )

    lhs_expr = parse_expression(lhs_str, local_symbols=local_symbols)
    rhs_expr = parse_expression(rhs_str, local_symbols=local_symbols)

    return Eq(lhs_expr, rhs_expr)


def extract_symbols(expr: Expr | Eq) -> set[Symbol]:
    """Extract all free symbols from a SymPy Expr or Eq."""
    return set(expr.free_symbols)


def extract_symbol_names(expr: Expr | Eq) -> list[str]:
    """Extract all free symbol names as sorted strings."""
    return sorted([str(s) for s in expr.free_symbols])
