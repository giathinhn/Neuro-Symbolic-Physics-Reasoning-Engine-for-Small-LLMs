"""Tests for ProblemClassifier distinguishing quantitative vs qualitative problems."""

from __future__ import annotations

import pytest

from physics_reasoning.core.enums import ProblemType
from physics_reasoning.pipeline.classifier import ProblemClassifier


class TestProblemClassifier:
    def test_classify_qualitative_vietnamese_inertia(self):
        text = "Tại sao khi xe buýt phanh gấp thì hành khách ngồi trên xe bị ngã chúi về phía trước?"
        assert ProblemClassifier.classify(text) == ProblemType.QUALITATIVE

    def test_classify_qualitative_vietnamese_pressure(self):
        text = "Hãy giải thích vì sao đầu đinh lại được làm nhọn còn móng nhà làm to bản?"
        assert ProblemClassifier.classify(text) == ProblemType.QUALITATIVE

    def test_classify_qualitative_english(self):
        text = "Why does a thick glass cup break more easily than a thin one when boiling water is poured into it?"
        assert ProblemClassifier.classify(text) == ProblemType.QUALITATIVE

    def test_classify_quantitative_fma(self):
        text = "A 2 kg object experiences a force of 10 N. Calculate its acceleration."
        assert ProblemClassifier.classify(text) == ProblemType.QUANTITATIVE

    def test_classify_quantitative_vietnamese(self):
        text = "Một vật khối lượng 5 kg chịu tác dụng của lực 20 N. Tính gia tốc của vật bằng bao nhiêu?"
        assert ProblemClassifier.classify(text) == ProblemType.QUANTITATIVE
