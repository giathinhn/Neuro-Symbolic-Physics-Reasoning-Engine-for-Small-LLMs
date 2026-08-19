"""Enumerations for the physics reasoning engine."""

from __future__ import annotations

from enum import StrEnum


class Difficulty(StrEnum):
    """Problem difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ProblemSource(StrEnum):
    """Source of a physics problem."""

    SYNTHETIC = "synthetic"
    MANUAL = "manual"
    EXTERNAL = "external"


class ErrorSeverity(StrEnum):
    """Severity of a verification error."""

    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ErrorType(StrEnum):
    """Type of verification error."""

    DIMENSION_MISMATCH = "dimension_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    ARITHMETIC_ERROR = "arithmetic_error"
    IMPOSSIBLE_VALUE = "impossible_value"
    INVALID_EQUATION = "invalid_equation"
    SUBSTITUTION_ERROR = "substitution_error"
    INCONSISTENT_SYSTEM = "inconsistent_system"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN_VARIABLE = "unknown_variable"
    MISSING_QUANTITY = "missing_quantity"


class QuantityRole(StrEnum):
    """Role of a quantity in a problem."""

    GIVEN = "given"
    TARGET = "target"
    INTERMEDIATE = "intermediate"


class PhysicsDomain(StrEnum):
    """Physics domain categories."""

    MECHANICS = "mechanics"
    ELECTRICITY = "electricity"
    THERMODYNAMICS = "thermodynamics"
    OPTICS = "optics"
    WAVES = "waves"


class PhysicsTopic(StrEnum):
    """Physics topic subcategories."""

    KINEMATICS = "kinematics"
    NEWTON_LAWS = "newton_laws"
    WORK_ENERGY = "work_energy"
    MOMENTUM = "momentum"
    DENSITY_PRESSURE = "density_pressure"
    OHMS_LAW = "ohms_law"
    ELECTRICAL_POWER = "electrical_power"
