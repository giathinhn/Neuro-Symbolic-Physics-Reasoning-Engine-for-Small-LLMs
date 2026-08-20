"""Integration tests for Qualitative Neuro-Symbolic Reasoning pipeline."""

from __future__ import annotations

import json
import pytest

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.llm.provider import MockLLMProvider
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.physics.qualitative_kb import QualitativeKnowledgeBase
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def qual_kb() -> QualitativeKnowledgeBase:
    kb = QualitativeKnowledgeBase("data/knowledge/qualitative")
    kb.load()
    return kb


@pytest.fixture
def quant_kb() -> KnowledgeBase:
    kb = KnowledgeBase("data/knowledge")
    kb.load()
    return kb


class TestQualitativePipelineIntegration:
    def test_solve_qualitative_problem_single_attempt(self, qual_kb, quant_kb):
        valid_response = json.dumps(
            {
                "problem_understanding": "Giải thích hiện tượng ngã chúi khi phanh gấp",
                "observed_phenomenon": "Hành khách ngã về trước",
                "core_principles": ["inertia_law"],
                "causal_chain": [
                    {
                        "step_number": 1,
                        "state_or_action": "Xe và người đang cùng chuyển động",
                        "physical_mechanism": "Khi phanh gấp, lực ma sát hãm xe và chân người dừng lại.",
                        "governing_principle": "inertia_law",
                    },
                    {
                        "step_number": 2,
                        "state_or_action": "Phần thân trên của người",
                        "physical_mechanism": "Do có quán tính nên phần thân trên tiếp tục duy trì vận tốc cũ chuyển động về phía trước.",
                        "governing_principle": "inertia_law",
                    },
                ],
                "conclusion": "Khi xe phanh gấp, do quán tính nên thân trên hành khách tiếp tục duy trì vận tốc cũ khiến người ngã chúi về trước.",
                "scientific_keywords": ["quán tính", "duy trì vận tốc"],
            }
        )

        mock_llm = MockLLMProvider(responses=[valid_response])
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(max_retries=3),
            llm_provider=mock_llm,
            knowledge_base=quant_kb,
            qualitative_knowledge_base=qual_kb,
        )

        problem_text = "Tại sao khi xe buýt đang chạy mà phanh gấp thì hành khách lại bị ngã chúi về phía trước?"
        solution = orchestrator.solve(problem_text)

        assert solution.is_qualitative
        assert solution.is_verified
        assert solution.num_attempts == 1
        assert "inertia_law" in solution.principles_applied
        assert solution.final_explanation is not None

    def test_solve_qualitative_repair_loop_on_misconception(self, qual_kb, quant_kb):
        # Attempt 1: Has misconception "lực quán tính đẩy"
        bad_response = json.dumps(
            {
                "problem_understanding": "Xe phanh gấp",
                "observed_phenomenon": "Người ngã",
                "core_principles": ["inertia_law"],
                "causal_chain": [
                    {
                        "step_number": 1,
                        "state_or_action": "Xe phanh",
                        "physical_mechanism": "Lực quán tính tác dụng đẩy người về phía trước.",
                        "governing_principle": "inertia_law",
                    },
                    {
                        "step_number": 2,
                        "state_or_action": "Kết quả",
                        "physical_mechanism": "Người bị ngã do lực quán tính đẩy.",
                        "governing_principle": "inertia_law",
                    },
                ],
                "conclusion": "Người bị ngã do lực quán tính đẩy mạnh.",
                "scientific_keywords": [],
            }
        )

        # Attempt 2: Repaired correct explanation
        good_response = json.dumps(
            {
                "problem_understanding": "Xe phanh gấp",
                "observed_phenomenon": "Người ngã",
                "core_principles": ["inertia_law"],
                "causal_chain": [
                    {
                        "step_number": 1,
                        "state_or_action": "Xe và chân người dừng lại do phanh",
                        "physical_mechanism": "Chân tiếp xúc sàn xe nên giảm vận tốc theo xe.",
                        "governing_principle": "inertia_law",
                    },
                    {
                        "step_number": 2,
                        "state_or_action": "Thân trên",
                        "physical_mechanism": "Do có quán tính nên thân trên tiếp tục duy trì vận tốc cũ tiến về phía trước.",
                        "governing_principle": "inertia_law",
                    },
                ],
                "conclusion": "Do có quán tính nên thân trên hành khách tiếp tục duy trì vận tốc cũ khiến người bị ngã chúi về trước.",
                "scientific_keywords": ["quán tính", "duy trì vận tốc"],
            }
        )

        mock_llm = MockLLMProvider(responses=[bad_response, good_response])
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(max_retries=3),
            llm_provider=mock_llm,
            knowledge_base=quant_kb,
            qualitative_knowledge_base=qual_kb,
        )

        problem_text = "Tại sao khi xe buýt đang chạy mà phanh gấp thì hành khách bị ngã chúi về phía trước?"
        solution = orchestrator.solve(problem_text)

        assert solution.is_qualitative
        assert solution.is_verified
        assert solution.num_attempts == 2
        assert "inertia_law" in solution.principles_applied
