"""Tests for equation retriever."""

from __future__ import annotations

import pytest

from physics_reasoning.physics.equation_retriever import EquationRetriever
from physics_reasoning.physics.knowledge_base import KnowledgeBase


@pytest.fixture
def retriever() -> EquationRetriever:
    kb = KnowledgeBase("data/knowledge")
    kb.load()
    return EquationRetriever(kb)


class TestEquationRetriever:
    def test_retrieve_by_quantities(self, retriever):
        eqs = retriever.retrieve_by_quantities(["force", "mass"])
        assert len(eqs) > 0
        eq_ids = [e.id for e in eqs]
        assert "newton2" in eq_ids

    def test_match_expression_exact(self, retriever):
        eq = retriever.match_expression("F = m * a")
        assert eq is not None
        assert eq.id == "newton2"

    def test_match_expression_algebraic(self, retriever):
        # Rearranged form F - m*a = 0
        eq = retriever.match_expression("m * a = F")
        assert eq is not None
        assert eq.id == "newton2"

    def test_suggest_equations(self, retriever):
        # Given mass and velocity, find kinetic energy
        eqs = retriever.suggest_equations(
            given_quantities=["mass", "velocity"],
            target_quantity="kinetic_energy",
        )
        assert len(eqs) > 0
        assert any(e.id == "ke_def" for e in eqs)
