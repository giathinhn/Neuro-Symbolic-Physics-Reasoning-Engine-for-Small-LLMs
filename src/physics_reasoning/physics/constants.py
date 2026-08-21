"""Standard physical constants used across physics problem domains."""

from __future__ import annotations

from typing import Final

# Standard acceleration due to gravity near Earth's surface (m/s^2)
STANDARD_GRAVITY: Final[float] = 9.8

# Speed of light in vacuum (m/s)
SPEED_OF_LIGHT: Final[float] = 299_792_458.0

# Universal gravitational constant (N * m^2 / kg^2)
GRAVITATIONAL_CONSTANT: Final[float] = 6.67430e-11

# Standard atmospheric pressure (Pa)
STANDARD_ATMOSPHERE_PA: Final[float] = 101_325.0

# Density of pure water at 4 degC (kg/m^3)
WATER_DENSITY_KG_M3: Final[float] = 1000.0


# Map of constant symbol or name to standard numerical SI value
PHYSICAL_CONSTANTS: dict[str, float] = {
    "g": STANDARD_GRAVITY,
    "gravity": STANDARD_GRAVITY,
    "c": SPEED_OF_LIGHT,
    "speed_of_light": SPEED_OF_LIGHT,
    "G": GRAVITATIONAL_CONSTANT,
    "rho_water": WATER_DENSITY_KG_M3,
    "P_atm": STANDARD_ATMOSPHERE_PA,
    "c_water": 4200.0,
}

# Map of constant symbol to standard SI unit string
PHYSICAL_CONSTANT_UNITS: dict[str, str] = {
    "g": "m/s**2",
    "gravity": "m/s**2",
    "c": "m/s",
    "speed_of_light": "m/s",
    "G": "N * m**2 / kg**2",
    "rho_water": "kg / m**3",
    "P_atm": "Pa",
    "c_water": "J / (kg * kelvin)",
}

