"""Error taxonomy and descriptions for the verification pipeline."""

from __future__ import annotations

from physics_reasoning.core.enums import ErrorSeverity, ErrorType

# Descriptive mapping of error types to categories and explanation templates
ERROR_TAXONOMY: dict[ErrorType, dict[str, str]] = {
    ErrorType.DIMENSION_MISMATCH: {
        "category": "Physics Error",
        "default_severity": ErrorSeverity.ERROR,
        "description": "Left-hand side and right-hand side of equation have incompatible physical dimensions.",
        "repair_hint": "Check the equation form and verify variable dimensions (e.g. force is M*L/T^2, energy is M*L^2/T^2).",
    },
    ErrorType.UNIT_MISMATCH: {
        "category": "Unit Error",
        "default_severity": ErrorSeverity.ERROR,
        "description": "Quantities have incompatible units or answer unit does not match requested target unit.",
        "repair_hint": "Convert all input quantities to standard SI units before applying formulas.",
    },
    ErrorType.ARITHMETIC_ERROR: {
        "category": "Calculation Error",
        "default_severity": ErrorSeverity.ERROR,
        "description": "Numerical calculation or expression evaluation failed.",
        "repair_hint": "Let the symbolic solver perform algebraic rearrangement and calculation.",
    },
    ErrorType.IMPOSSIBLE_VALUE: {
        "category": "Physical Bound Error",
        "default_severity": ErrorSeverity.ERROR,
        "description": "Computed value violates physical laws or domain constraints (e.g. negative mass, speed > c).",
        "repair_hint": "Verify signs, initial values, and check for square root sign ambiguities.",
    },
    ErrorType.INVALID_EQUATION: {
        "category": "Semantic Error",
        "default_severity": ErrorSeverity.ERROR,
        "description": "The proposed equation is structurally invalid or does not exist in standard physics.",
        "repair_hint": "Select applicable formulas from standard physics laws (e.g., F=ma, v=d/t, KE=1/2*m*v^2).",
    },
    ErrorType.SUBSTITUTION_ERROR: {
        "category": "Verification Error",
        "default_severity": ErrorSeverity.ERROR,
        "description": "Substituting calculated answers back into the original equations produces a non-zero residual.",
        "repair_hint": "Recheck equation formulation and known value assignment.",
    },
    ErrorType.INCONSISTENT_SYSTEM: {
        "category": "System Consistency Error",
        "default_severity": ErrorSeverity.FATAL,
        "description": "The system of equations has contradictory constraints and no mathematical solution.",
        "repair_hint": "Eliminate conflicting equation assumptions or re-examine problem conditions.",
    },
    ErrorType.SYNTAX_ERROR: {
        "category": "Syntax Error",
        "default_severity": ErrorSeverity.ERROR,
        "description": "Equation or expression cannot be parsed.",
        "repair_hint": "Ensure equations use standard notation with exactly one '=' sign.",
    },
    ErrorType.MISSING_QUANTITY: {
        "category": "Completeness Error",
        "default_severity": ErrorSeverity.WARNING,
        "description": "A required variable in the equation is not provided in the problem statement.",
        "repair_hint": "Check for implicit quantities (e.g. 'from rest' -> v_i = 0, 'on Earth' -> g = 9.8 m/s^2).",
    },
}
