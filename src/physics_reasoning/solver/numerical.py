"""Numerical utilities and physical domain reasonableness checks."""

from __future__ import annotations

import math
from typing import Any

import sympy

# Physical constants for bounds checking
SPEED_OF_LIGHT = 299_792_458.0  # m/s
ABSOLUTE_ZERO_CELSIUS = -273.15  # degC
ABSOLUTE_ZERO_KELVIN = 0.0  # K


def evaluate_numeric(val: Any, precision: int = 10) -> float:
    """Convert a SymPy expression or numeric object to a Python float.

    Args:
        val: SymPy number, expression, int, float, or str.
        precision: Precision for SymPy evalf.

    Returns:
        float representation.

    Raises:
        ValueError: If value cannot be evaluated to a real float.
    """
    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, sympy.Basic):
        evalf_val = val.evalf(precision)
        if evalf_val.is_real is False:
            raise ValueError(f"Value '{val}' is complex/non-real: {evalf_val}")
        return float(evalf_val)

    try:
        return float(val)
    except Exception as e:
        raise ValueError(f"Cannot convert '{val}' of type {type(val)} to float: {e}") from e


def round_to_significant_figures(value: float, sig_figs: int = 4) -> float:
    """Round a float to a specific number of significant figures.

    Args:
        value: The numerical value.
        sig_figs: Number of significant figures (must be >= 1).

    Returns:
        Rounded float.
    """
    if sig_figs < 1:
        raise ValueError("sig_figs must be >= 1")
    if value == 0.0 or not math.isfinite(value):
        return value

    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (sig_figs - 1 - magnitude)
    return round(value * factor) / factor


def is_close(
    a: float | None,
    b: float | None,
    rel_tol: float = 0.01,
    abs_tol: float = 1e-6,
) -> bool:
    """Check if two floats are close within relative and absolute tolerance.

    Args:
        a: First number.
        b: Second number.
        rel_tol: Relative tolerance (default 1%).
        abs_tol: Absolute tolerance for values near zero.

    Returns:
        True if close, False otherwise.
    """
    if a is None or b is None:
        return False
    if not math.isfinite(a) or not math.isfinite(b):
        return False

    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def is_physically_reasonable(
    value: float, quantity_name: str
) -> tuple[bool, str | None]:
    """Check if a calculated or proposed value is physically plausible.

    Args:
        value: Numerical value (assumed in SI units).
        quantity_name: Name or alias of physical quantity.

    Returns:
        (is_reasonable, reason_if_not)
    """
    if not math.isfinite(value):
        return False, f"Value {value} is not finite (NaN or Inf)"

    q = quantity_name.lower().strip()

    # Mass must be strictly positive
    if q in ("mass", "m", "object_mass"):
        if value <= 0:
            return False, f"Mass must be positive, got {value}"

    # Speeds must be non-negative (for scalar speed) and strictly < speed of light
    if q in ("speed", "velocity", "v", "initial_velocity", "final_velocity", "v_i", "v_f"):
        if abs(value) >= SPEED_OF_LIGHT:
            return False, f"Speed {abs(value)} m/s exceeds speed of light ({SPEED_OF_LIGHT} m/s)"

    # Time durations must typically be non-negative
    if q in ("time", "t", "duration", "time_interval"):
        if value < 0:
            return False, f"Time interval cannot be negative, got {value}"

    # Absolute temperature in Kelvin must be >= 0
    if q in ("temperature_k", "kelvin", "t_kelvin"):
        if value < ABSOLUTE_ZERO_KELVIN:
            return False, f"Temperature in Kelvin cannot be below 0 K, got {value} K"

    # Resistance must be non-negative
    if q in ("resistance", "r"):
        if value < 0:
            return False, f"Resistance cannot be negative, got {value}"

    # Frequency must be non-negative
    if q in ("frequency", "freq", "f"):
        if value < 0:
            return False, f"Frequency cannot be negative, got {value}"

    # Kinetic energy must be non-negative
    if q in ("kinetic_energy", "ke"):
        if value < 0:
            return False, f"Kinetic energy cannot be negative, got {value}"

    return True, None
