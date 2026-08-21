import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("src"))

from physics_reasoning.units.dimension_checker import DimensionChecker

dc = DimensionChecker()
# What var_units might be passed:
# e.g. {'U': 'V', 'R': 'ohm', 'I': 'A'} or {'c': '', 'U': 'V', 'R': 'ohm'}
print("Test 1:", dc.check_equation("I = U / R", {"U": "V", "R": "ohm", "I": "A"}))
print("Test 2:", dc.check_equation("I = U / R", {"U": "V", "R": "ohm", "I": "current"}))
print("Test 3:", dc.check_equation("I = U / R", {"U": "V", "R": "ohm"}))
