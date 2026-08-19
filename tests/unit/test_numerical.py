"""Tests for numerical utilities."""

from __future__ import annotations

import math
import pytest
import sympy

from physics_reasoning.solver.numerical import (
    evaluate_numeric,
    is_close,
    is_physically_reasonable,
    round_to_significant_figures,
)


class TestNumericalUtilities:
    def test_evaluate_numeric_int_float(self):
        assert evaluate_numeric(5) == 5.0
        assert evaluate_numeric(3.14) == 3.14

    def test_evaluate_numeric_sympy(self):
        sym_expr = sympy.sqrt(16)
        assert evaluate_numeric(sym_expr) == 4.0

    def test_evaluate_numeric_fraction(self):
        sym_expr = sympy.Rational(1, 2)
        assert evaluate_numeric(sym_expr) == 0.5

    def test_round_to_significant_figures(self):
        assert round_to_significant_figures(1234.56, 3) == 1230.0
        assert round_to_significant_figures(0.0012345, 2) == 0.0012
        assert round_to_significant_figures(5.0, 4) == 5.0
        assert round_to_significant_figures(0.0, 3) == 0.0

    def test_is_close(self):
        assert is_close(5.0, 5.0001, rel_tol=0.01)
        assert not is_close(5.0, 6.0, rel_tol=0.01)
        assert is_close(0.0, 1e-7, abs_tol=1e-6)
        assert not is_close(None, 5.0)

    def test_is_physically_reasonable_mass(self):
        ok, _ = is_physically_reasonable(10.0, "mass")
        assert ok
        ok, msg = is_physically_reasonable(-2.0, "mass")
        assert not ok
        assert "positive" in msg

    def test_is_physically_reasonable_speed(self):
        ok, _ = is_physically_reasonable(30.0, "speed")
        assert ok
        ok, msg = is_physically_reasonable(3e9, "speed")
        assert not ok
        assert "speed of light" in msg

    def test_is_physically_reasonable_time(self):
        ok, _ = is_physically_reasonable(5.0, "time")
        assert ok
        ok, msg = is_physically_reasonable(-5.0, "time")
        assert not ok
        assert "negative" in msg
