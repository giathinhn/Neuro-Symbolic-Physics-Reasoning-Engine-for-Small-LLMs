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
        assert len(qual_kb.principles) >= 15
        assert "inertia_law" in qual_kb.principles
        assert "solid_pressure_area" in qual_kb.principles
        assert "thermal_expansion_unequal" in qual_kb.principles
        assert "condensation_dew_formation" in qual_kb.principles
        assert "light_refraction_phenomena" in qual_kb.principles
        assert "sound_propagation_media" in qual_kb.principles

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

    def test_find_relevant_principles_condensation(self, qual_kb):
        results = qual_kb.find_relevant_principles("Tại sao mặt ngoài của cốc nước đá lại có các giọt nước đọng?")
        assert len(results) >= 1
        assert results[0].id == "condensation_dew_formation"

    def test_find_relevant_principles_refraction(self, qual_kb):
        results = qual_kb.find_relevant_principles("Cắm chiếc đũa vào cốc nước thấy đũa như bị gãy khúc")
        assert len(results) >= 1
        assert results[0].id == "light_refraction_phenomena"

    def test_find_relevant_principles_sound_vacuum(self, qual_kb):
        results = qual_kb.find_relevant_principles("Tại sao phi hành gia trong không gian không nói chuyện trực tiếp được?")
        assert len(results) >= 1
        assert results[0].id == "sound_propagation_media"

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

    def test_scan_for_misconception_water_leak(self, qual_kb):
        bad_text = "Nước trong cốc ngấm qua thành cốc ra ngoài làm ướt mặt ngoài."
        matches = qual_kb.scan_for_misconceptions(bad_text)
        assert len(matches) >= 1
        assert matches[0]["id"] == "cup_sweats_water_leak"

    def test_scan_for_clean_explanation_no_misconception(self, qual_kb):
        good_text = (
            "Do có quán tính nên phần thân trên tiếp tục duy trì vận tốc cũ, "
            "trong khi phần dưới dừng lại cùng xe."
        )
        matches = qual_kb.scan_for_misconceptions(good_text)
        assert len(matches) == 0

    def test_synthesize_fallback_explanation(self, qual_kb):
        p = qual_kb.get_principle("condensation_dew_formation")
        assert p is not None
        out = qual_kb.synthesize_fallback_explanation(
            "Tại sao mặt ngoài cốc nước đá bị đọng giọt nước?", p
        )
        assert len(out.causal_chain) >= 2
        assert len(out.conclusion) > 20
        assert "ngưng tụ" in out.conclusion.lower() or "condensation" in out.core_principles[0]
