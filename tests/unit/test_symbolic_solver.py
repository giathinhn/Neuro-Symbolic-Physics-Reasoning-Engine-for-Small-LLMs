"""Tests for symbolic solver."""

from __future__ import annotations

import math
import pytest

from physics_reasoning.solver.symbolic_solver import SymbolicSolver


@pytest.fixture
def solver() -> SymbolicSolver:
    return SymbolicSolver()


class TestSymbolicSolver:
    def test_solve_fma_for_acceleration(self, solver):
        res = solver.solve_single("F = m * a", {"F": 10.0, "m": 2.0}, "a")
        assert res.is_numeric
        assert len(res.solutions) == 1
        assert math.isclose(res.solutions[0], 5.0)

    def test_solve_fma_for_force(self, solver):
        res = solver.solve_single("F = m * a", {"m": 2.5, "a": 4.0}, "F")
        assert res.is_numeric
        assert math.isclose(res.solutions[0], 10.0)

    def test_solve_fma_for_mass(self, solver):
        res = solver.solve_single("F = m * a", {"F": 50.0, "a": 5.0}, "m")
        assert res.is_numeric
        assert math.isclose(res.solutions[0], 10.0)

    def test_solve_kinematics_v_eq_d_div_t(self, solver):
        res = solver.solve_single("v = d / t", {"d": 100.0, "t": 5.0}, "v")
        assert res.is_numeric
        assert math.isclose(res.solutions[0], 20.0)

    def test_solve_quadratic_kinetic_energy(self, solver):
        res = solver.solve_single("KE = (1/2) * m * v^2", {"KE": 100.0, "m": 2.0}, "v")
        assert res.is_numeric
        # Should contain 10.0 or -10.0
        assert any(math.isclose(s, 10.0) for s in res.solutions)

    def test_solve_system_two_equations(self, solver):
        eqs = [
            "F_net = F1 - F2",
            "F_net = m * a",
        ]
        knowns = {"F1": 20.0, "F2": 5.0, "m": 3.0}
        res = solver.solve_system(eqs, knowns, "a")
        assert res.is_numeric
        assert math.isclose(res.solutions[0], 5.0)

    def test_solve_no_solution_inconsistent(self, solver):
        # x = 1 and x = 2
        eqs = ["x = 1", "x = 2"]
        res = solver.solve_system(eqs, {}, "x")
        assert not res.is_numeric or len(res.solutions) == 0

    def test_verify_substitution_valid(self, solver):
        is_sat, res = solver.verify_substitution("F = m * a", {"F": 10.0, "m": 2.0, "a": 5.0})
        assert is_sat
        assert math.isclose(res, 0.0, abs_tol=1e-5)

    def test_verify_substitution_invalid(self, solver):
        is_sat, res = solver.verify_substitution("F = m * a", {"F": 10.0, "m": 2.0, "a": 99.0})
        assert not is_sat
        assert res > 1.0
