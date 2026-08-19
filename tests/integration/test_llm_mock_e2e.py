"""End-to-end tests verifying multi-equation and edge case problems."""

from __future__ import annotations

import json
import math
import pytest

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.llm.provider import MockLLMProvider
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def kb() -> KnowledgeBase:
    k = KnowledgeBase("data/knowledge")
    k.load()
    return k


class TestEndToEndMock:
    def test_multi_equation_energy_conservation(self, kb):
        # 5 kg object dropped from 3 meters. Find final velocity (v = sqrt(2*g*h) = sqrt(2*9.8*3) = 7.668 m/s)
        resp = json.dumps(
            {
                "problem_understanding": "Speed at bottom of frictionless ramp via energy conservation",
                "quantities": [
                    {"name": "mass", "symbol": "m", "value": 5.0, "unit": "kg", "role": "given"},
                    {"name": "height", "symbol": "h", "value": 3.0, "unit": "m", "role": "given"},
                    {"name": "velocity", "symbol": "v", "unit": "m/s", "role": "target"},
                ],
                "equations": [
                    {"equation_id": "energy_cons", "expression": "(1/2) * m * v^2 = m * g * h"}
                ],
                "target_variable": "v",
                "proposed_unit": "m/s",
            }
        )
        mock_llm = MockLLMProvider([resp])
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(max_retries=2),
            llm_provider=mock_llm,
            knowledge_base=kb,
        )

        sol = orchestrator.solve("A 5 kg block slides down a ramp from height 3 m. Find speed at bottom.")
        assert sol.is_verified
        expected = math.sqrt(2 * 9.8 * 3.0)
        assert math.isclose(sol.answer_value, expected, rel_tol=0.01)

    def test_unsolvable_exceeds_retries_returns_unverified(self, kb):
        # Model continually returns garbage equations
        bad_resp = json.dumps(
            {
                "quantities": [{"name": "mass", "symbol": "m", "value": 5.0}],
                "equations": [{"expression": "F = m * t"}],
                "target_variable": "F",
            }
        )
        mock_llm = MockLLMProvider([bad_resp, bad_resp, bad_resp])
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(max_retries=2),
            llm_provider=mock_llm,
            knowledge_base=kb,
        )

        sol = orchestrator.solve("Invalid physics problem.")
        assert not sol.is_verified
        assert sol.num_attempts == 2
