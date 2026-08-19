"""Regression test suite for previously-discovered physics reasoning bugs and edge cases."""

from __future__ import annotations

import json
import math
import pytest

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.core.models import Dimension
from physics_reasoning.llm.provider import MockLLMProvider
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator
from physics_reasoning.solver.expression_parser import parse_equation_string, parse_expression
from physics_reasoning.solver.symbolic_solver import SymbolicSolver
from physics_reasoning.units.dimension_checker import DimensionChecker
from physics_reasoning.units.unit_engine import UnitEngine


@pytest.fixture
def kb() -> KnowledgeBase:
    k = KnowledgeBase("data/knowledge")
    k.load()
    return k


class TestRegressionFailures:
    def test_theta_temperature_dimension_regex(self):
        """Regression: Dimension.from_string must match Theta without greedily matching T."""
        d = Dimension.from_string("Theta")
        assert d == Dimension(Theta=1)
        assert d.T == 0

    def test_indexed_variables_not_split(self):
        """Regression: Variables with digits like F1, F2, v1, m2 must parse as single symbols."""
        eq = parse_equation_string("F_net = F1 - F2")
        symbols = [str(s) for s in eq.free_symbols]
        assert "F1" in symbols
        assert "F2" in symbols
        assert "F_net" in symbols

    def test_ke_not_split_into_k_times_euler_e(self):
        """Regression: Kinetic energy KE must parse as Symbol('KE') not K * E."""
        expr = parse_expression("KE")
        assert str(expr) == "KE"
        assert expr.is_symbol

    def test_implicit_constant_gravity_units_in_dimensional_check(self):
        """Regression: Gravitational potential energy PE = m * g * h must be dimensionally consistent with implicit g."""
        dim_checker = DimensionChecker()
        res = dim_checker.check_equation(
            "PE = m * g * h",
            {"PE": "J", "m": "kg", "h": "m"},  # 'g' is implicit constant
        )
        assert res.is_consistent

    def test_quadratic_kinetic_energy_solver_root_selection(self, kb):
        """Regression: Solving for velocity in KE = (1/2)*m*v^2 must produce positive physical root."""
        solver = SymbolicSolver()
        res = solver.solve_single("KE = (1/2) * m * v^2", {"KE": 100.0, "m": 2.0}, "v")
        assert res.is_numeric
        assert any(math.isclose(s, 10.0) for s in res.solutions)
