"""Expression parser for converting string equations/math to SymPy objects safely."""

from __future__ import annotations

import re
from typing import Any

import sympy
from sympy import Eq, Expr, Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from physics_reasoning.core.exceptions import ExpressionParseError

# Standard safe transformations for physics expressions
SAFE_TRANSFORMATIONS = standard_transformations + (convert_xor,)

# Whitelist of allowed SymPy mathematical functions and constants
# Note: 'E' is omitted because in physics E denotes energy, not Euler's number
ALLOWED_FUNCTIONS: dict[str, Any] = {
    "pi": sympy.pi,
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


def _preprocess_math_string(s: str) -> str:
    """Normalize physics math strings before parsing.

    - Convert implicit numeric multiplications: '2 a' -> '2 * a', '2(x)' -> '2 * (x)'
    - Preserve variable names with digits: 'F1', 'm2', 'v_i'
    - Handle '^' as power (handled by convert_xor)
    """
    # Replace number followed by space and identifier: '2 a' -> '2 * a'
    s = re.sub(r"(\d+(?:\.\d+)?)\s+([a-zA-Z_][a-zA-Z0-9_]*)", r"\1 * \2", s)
    # Replace number immediately followed by parenthesis: '2(a + b)' -> '2 * (a + b)'
    s = re.sub(r"(\d+(?:\.\d+)?)\s*\(", r"\1 * (", s)
    return s


def parse_expression(
    expr_str: str, local_symbols: dict[str, Symbol] | None = None
) -> Expr:
    """Parse a string into a SymPy expression safely.

    Features:
    - Whitelisted math functions (sqrt, sin, cos, etc.)
    - Multi-letter and indexed physics variables (KE, PE, F1, F2, v_i, F_net)
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

    preprocessed = _preprocess_math_string(clean_str)

    # Extract all identifier tokens
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", preprocessed)
    local_dict: dict[str, Any] = dict(ALLOWED_FUNCTIONS)

    # Any identifier that is not an allowed math function is treated as a Symbol
    for token in tokens:
        if token not in ALLOWED_FUNCTIONS:
            local_dict[token] = Symbol(token)

    if local_symbols:
        local_dict.update(local_symbols)

    try:
        parsed = parse_expr(
            preprocessed,
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

    return Eq(lhs_expr, rhs_expr, evaluate=False)


def extract_symbols(expr: Expr | Eq | str) -> set[Symbol]:
    """Extract all free symbols from a SymPy Expr, Eq, or equation string."""
    if isinstance(expr, str):
        if "=" in expr:
            parsed = parse_equation_string(expr)
        else:
            parsed = parse_expression(expr)
        return set(parsed.free_symbols)
    return set(expr.free_symbols)


def extract_symbol_names(expr: Expr | Eq | str) -> list[str]:
    """Extract all free symbol names as sorted strings."""
    try:
        if isinstance(expr, str):
            if "=" in expr:
                parsed = parse_equation_string(expr)
            else:
                parsed = parse_expression(expr)
            return sorted([str(s) for s in parsed.free_symbols])
        return sorted([str(s) for s in expr.free_symbols])
    except Exception:
        if isinstance(expr, str):
            tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr)
            return sorted(list(set(tokens) - set(ALLOWED_FUNCTIONS)))
        return []
