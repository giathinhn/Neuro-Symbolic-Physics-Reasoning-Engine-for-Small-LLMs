"""Tests for physics knowledge base manager."""

from __future__ import annotations

import pytest

from physics_reasoning.physics.knowledge_base import KnowledgeBase


@pytest.fixture
def kb() -> KnowledgeBase:
    k = KnowledgeBase("data/knowledge")
    k.load()
    return k


class TestKnowledgeBase:
    def test_load_quantities(self, kb):
        assert len(kb.quantities) >= 20
        force = kb.get_quantity_by_name("force")
        assert force is not None
        assert force.symbol == "F"
        assert force.dimension == "M L T^-2"

    def test_get_quantity_by_symbol(self, kb):
        mass = kb.get_quantity_by_symbol("m")
        assert mass is not None
        assert mass.name == "mass"

    def test_get_quantity_by_alias(self, kb):
        speed = kb.get_quantity_by_name("speed")
        assert speed is not None
        assert speed.name == "velocity"

    def test_load_equations(self, kb):
        assert len(kb.equations) >= 20
        newton2 = kb.get_equation("newton2")
        assert newton2 is not None
        assert newton2.expression == "F = m * a"

    def test_search_by_quantities(self, kb):
        matches = kb.search_by_quantities(["force", "mass", "acceleration"])
        assert len(matches) > 0
        assert matches[0].id == "newton2"

    def test_search_by_topic(self, kb):
        kin_eqs = kb.search_by_topic("kinematics")
        assert len(kin_eqs) >= 5
        assert any(eq.id == "kin_vel_def" for eq in kin_eqs)

    def test_validate_no_warnings(self, kb):
        warnings = kb.validate()
        assert len(warnings) == 0
