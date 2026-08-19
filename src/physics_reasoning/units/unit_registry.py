"""Customized Pint UnitRegistry and aliases for physics quantities."""

from __future__ import annotations

import pint
from pint import UnitRegistry

# Singleton unit registry instance
_UREG: UnitRegistry | None = None


def get_unit_registry() -> UnitRegistry:
    """Get or create the global Pint UnitRegistry with custom aliases."""
    global _UREG
    if _UREG is None:
        _UREG = UnitRegistry(autoconvert_offset_to_baseunit=True)
        # Define any additional aliases if needed
        # Pint already has N, J, W, Pa, m, s, kg, km, h, etc.
    return _UREG
