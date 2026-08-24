"""Tests for Qualitative Verification Pipeline and individual checks."""

from __future__ import annotations

import pytest

from physics_reasoning.core.enums import ErrorType
from physics_reasoning.core.models import CausalStep, QualitativeParsedOutput
from physics_reasoning.physics.qualitative_kb import QualitativeKnowledgeBase
from physics_reasoning.verifier.qualitative_verifier import (
    AntiTautologyCheck,
    CausalCompletenessCheck,
    CausalDirectionalityCheck,
    MisconceptionCheck,
    PrincipleRelevanceCheck,
    QualitativeVerificationPipeline,
    ScientificTerminologyCheck,
)


@pytest.fixture
def qual_kb() -> QualitativeKnowledgeBase:
    kb = QualitativeKnowledgeBase("data/knowledge/qualitative")
    kb.load()
    return kb


@pytest.fixture
def valid_inertia_output() -> QualitativeParsedOutput:
    return QualitativeParsedOutput(
        problem_understanding="Xe phanh gấp làm người ngã về phía trước",
        observed_phenomenon="Người ngã chúi về trước",
        core_principles=["inertia_law"],
        causal_chain=[
            CausalStep(
                step_number=1,
                state_or_action="Xe và người đang cùng chuyển động về trước.",
                physical_mechanism="Khi phanh gấp, lực ma sát hãm xe và chân người dừng lại.",
                governing_principle="inertia_law",
            ),
            CausalStep(
                step_number=2,
                state_or_action="Phần thân trên chưa chịu lực hãm trực tiếp.",
                physical_mechanism="Do có quán tính nên phần thân trên tiếp tục duy trì vận tốc cũ về phía trước.",
                governing_principle="inertia_law",
            ),
            CausalStep(
                step_number=3,
                state_or_action="Kết quả tương đối giữa hai phần cơ thể.",
                physical_mechanism="Thân trên chuyển động nhanh hơn thân dưới làm người ngã chúi về phía trước.",
                governing_principle="inertia_law",
            ),
        ],
        conclusion="Khi xe phanh gấp, do có quán tính nên thân trên tiếp tục duy trì vận tốc cũ làm người ngã chúi về phía trước.",
        scientific_keywords=["quán tính", "duy trì vận tốc"],
    )


class TestQualitativeVerifier:
    def test_principle_relevance_check_pass(self, qual_kb, valid_inertia_output):
        check = PrincipleRelevanceCheck(qual_kb)
        errs = check.run("Tại sao khi xe phanh gấp người ngã chúi?", valid_inertia_output)
        assert len(errs) == 0

    def test_principle_relevance_check_no_principles(self, qual_kb, valid_inertia_output):
        bad = valid_inertia_output.model_copy(update={"core_principles": []})
        check = PrincipleRelevanceCheck(qual_kb)
        errs = check.run("Tại sao khi xe phanh gấp?", bad)
        assert len(errs) > 0
        assert errs[0].error_type == ErrorType.INVALID_EQUATION

    def test_causal_completeness_check_pass(self, valid_inertia_output):
        check = CausalCompletenessCheck()
        errs = check.run("Tại sao xe phanh gấp?", valid_inertia_output)
        assert len(errs) == 0

    def test_causal_completeness_check_too_short(self, valid_inertia_output):
        bad = valid_inertia_output.model_copy(update={"causal_chain": [valid_inertia_output.causal_chain[0]]})
        check = CausalCompletenessCheck()
        errs = check.run("Tại sao xe phanh gấp?", bad)
        assert len(errs) > 0

    def test_misconception_check_detects_force_fallacy(self, qual_kb, valid_inertia_output):
        bad_chain = [
            CausalStep(
                step_number=1,
                state_or_action="Xe phanh",
                physical_mechanism="Lực quán tính tác dụng đẩy người về phía trước.",
            ),
            CausalStep(
                step_number=2,
                state_or_action="Kết quả",
                physical_mechanism="Người bị ngã do lực quán tính đẩy mạnh.",
            ),
        ]
        bad = valid_inertia_output.model_copy(
            update={
                "causal_chain": bad_chain,
                "conclusion": "Người bị ngã do lực quán tính đẩy mạnh về phía trước.",
            }
        )
        check = MisconceptionCheck(qual_kb)
        errs = check.run("Xe phanh gấp", bad)
        assert len(errs) > 0
        assert "ngộ nhận" in errs[0].message.lower()

    def test_causal_directionality_check_detects_wrong_direction(self, valid_inertia_output):
        bad = valid_inertia_output.model_copy(
            update={
                "conclusion": "Khi xe phanh gấp, người ngã ra phía sau.",
            }
        )
        check = CausalDirectionalityCheck()
        errs = check.run("Tại sao khi xe phanh gấp người ngã chúi?", bad)
        assert len(errs) > 0
        assert "phía trước" in errs[0].message.lower()

    def test_anti_tautology_check_detects_circular_reasoning(self, valid_inertia_output):
        bad = valid_inertia_output.model_copy(
            update={
                "conclusion": "Hiện tượng này xảy ra vì hiện tượng này là như vậy.",
            }
        )
        check = AntiTautologyCheck()
        errs = check.run("Tại sao khi xe phanh gấp?", bad)
        assert len(errs) > 0
        assert "lặp luận" in errs[0].message.lower() or "tautology" in errs[0].message.lower()

    def test_full_qualitative_pipeline_pass(self, qual_kb, valid_inertia_output):
        pipe = QualitativeVerificationPipeline(qual_kb)
        res = pipe.verify("Tại sao khi xe phanh gấp người ngã về phía trước?", valid_inertia_output)
        assert res.is_valid
        assert res.confidence == 1.0
        assert len(res.errors) == 0
