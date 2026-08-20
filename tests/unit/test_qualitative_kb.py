"""Tests for QualitativeKnowledgeBase and misconception scanner."""

from __future__ import annotations

import pytest

from physics_reasoning.physics.qualitative_kb import QualitativeKnowledgeBase


@pytest.fixture
def qual_kb() -> QualitativeKnowledgeBase:
    kb = QualitativeKnowledgeBase("data/knowledge/qualitative")
    kb.load()
    return kb


class TestQualitativeKB:
    def test_load_principles(self, qual_kb):
        assert len(qual_kb.principles) >= 7
        assert "inertia_law" in qual_kb.principles
        assert "solid_pressure_area" in qual_kb.principles
        assert "thermal_expansion_unequal" in qual_kb.principles

    def test_find_relevant_principles_inertia(self, qual_kb):
        results = qual_kb.find_relevant_principles("Tại sao khi xe phanh gấp hành khách bị ngã chúi?")
        assert len(results) >= 1
        assert results[0].id == "inertia_law"

    def test_find_relevant_principles_pressure(self, qual_kb):
        results = qual_kb.find_relevant_principles("Tại sao đầu đinh lại được làm nhọn còn móng nhà làm to bản?")
        assert len(results) >= 1
        assert results[0].id == "solid_pressure_area"

    def test_find_relevant_principles_thermal_expansion(self, qual_kb):
        results = qual_kb.find_relevant_principles("Rót nước sôi vào cốc thủy tinh dày dễ vỡ hơn cốc mỏng")
        assert len(results) >= 1
        assert results[0].id == "thermal_expansion_unequal"

    def test_scan_for_misconception_inertia_force(self, qual_kb):
        bad_text = "Khi xe phanh gấp, có lực quán tính đẩy người về phía trước."
        matches = qual_kb.scan_for_misconceptions(bad_text)
        assert len(matches) >= 1
        assert matches[0]["id"] == "inertia_as_force"

    def test_scan_for_misconception_nail_force(self, qual_kb):
        bad_text = "Đầu đinh nhọn làm tăng lực tác dụng vào gỗ nên dễ đóng hơn."
        matches = qual_kb.scan_for_misconceptions(bad_text)
        assert len(matches) >= 1
        assert matches[0]["id"] == "nail_sharp_force"

    def test_scan_for_clean_explanation_no_misconception(self, qual_kb):
        good_text = (
            "Do có quán tính nên phần thân trên tiếp tục duy trì vận tốc cũ, "
            "trong khi phần dưới dừng lại cùng xe."
        )
        matches = qual_kb.scan_for_misconceptions(good_text)
        assert len(matches) == 0
