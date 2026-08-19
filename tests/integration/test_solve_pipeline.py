"""Integration tests for pipeline orchestrator."""

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


class TestSolvePipelineIntegration:
    def test_solve_fma_problem_single_attempt(self, kb):
        mock_response = json.dumps(
            {
                "problem_understanding": "Calculate acceleration given mass and force",
                "quantities": [
                    {"name": "mass", "symbol": "m", "value": 2.0, "unit": "kg", "role": "given"},
                    {"name": "force", "symbol": "F", "value": 10.0, "unit": "N", "role": "given"},
                    {"name": "acceleration", "symbol": "a", "unit": "m/s**2", "role": "target"},
                ],
                "equations": [
                    {"equation_id": "newton2", "expression": "F = m * a"}
                ],
                "target_variable": "a",
                "solution_steps": ["F = m * a", "a = F / m", "a = 10 / 2 = 5"],
                "proposed_unit": "m/s**2",
            }
        )
        mock_llm = MockLLMProvider([mock_response])
        config = PipelineConfig(max_retries=2)
        orchestrator = PipelineOrchestrator(
            config=config,
            llm_provider=mock_llm,
            knowledge_base=kb,
        )

        problem_text = "A 2 kg object experiences a force of 10 N. Find its acceleration."
        solution = orchestrator.solve(problem_text)

        assert solution.is_verified
        assert solution.answer_value is not None
        assert math.isclose(solution.answer_value, 5.0)
        assert solution.answer_unit in ("m/s**2", "meter / second ** 2")
        assert solution.num_attempts == 1
        assert solution.total_llm_calls == 1

    def test_solve_with_unit_conversion(self, kb):
        # 72 km/h deceleration at 4 m/s^2 -> t = 5 s
        mock_response = json.dumps(
            {
                "problem_understanding": "Car stopping with unit conversion",
                "quantities": [
                    {"name": "initial_velocity", "symbol": "v_i", "value": 72.0, "unit": "km/h", "role": "given"},
                    {"name": "final_velocity", "symbol": "v_f", "value": 0.0, "unit": "m/s", "role": "given"},
                    {"name": "acceleration", "symbol": "a", "value": -4.0, "unit": "m/s**2", "role": "given"},
                    {"name": "time", "symbol": "t", "unit": "s", "role": "target"},
                ],
                "equations": [
                    {"equation_id": "kin_eq1", "expression": "v_f = v_i + a * t"}
                ],
                "target_variable": "t",
                "proposed_unit": "s",
            }
        )
        mock_llm = MockLLMProvider([mock_response])
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(max_retries=2),
            llm_provider=mock_llm,
            knowledge_base=kb,
        )

        solution = orchestrator.solve("Car at 72 km/h stops with deceleration 4 m/s^2. Find time.")
        assert solution.is_verified
        assert solution.answer_value is not None
        assert math.isclose(solution.answer_value, 5.0)

    def test_solve_repair_loop_on_wrong_equation(self, kb):
        # Attempt 1 has wrong equation (F = m * t) -> fails dimension check
        bad_response = json.dumps(
            {
                "problem_understanding": "Wrong formula attempt",
                "quantities": [
                    {"name": "mass", "symbol": "m", "value": 2.0, "unit": "kg", "role": "given"},
                    {"name": "force", "symbol": "F", "value": 10.0, "unit": "N", "role": "given"},
                    {"name": "acceleration", "symbol": "a", "unit": "m/s**2", "role": "target"},
                ],
                "equations": [
                    {"expression": "F = m * t"}
                ],
                "target_variable": "a",
            }
        )
        # Attempt 2 provides corrected equation (F = m * a)
        good_response = json.dumps(
            {
                "problem_understanding": "Corrected formula attempt",
                "quantities": [
                    {"name": "mass", "symbol": "m", "value": 2.0, "unit": "kg", "role": "given"},
                    {"name": "force", "symbol": "F", "value": 10.0, "unit": "N", "role": "given"},
                    {"name": "acceleration", "symbol": "a", "unit": "m/s**2", "role": "target"},
                ],
                "equations": [
                    {"equation_id": "newton2", "expression": "F = m * a"}
                ],
                "target_variable": "a",
                "proposed_unit": "m/s**2",
            }
        )
        mock_llm = MockLLMProvider([bad_response, good_response])
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(max_retries=3),
            llm_provider=mock_llm,
            knowledge_base=kb,
        )

        solution = orchestrator.solve("A 2 kg object experiences a force of 10 N. Find acceleration.")
        assert solution.is_verified
        assert solution.num_attempts == 2
        assert solution.total_llm_calls == 2
        assert math.isclose(solution.answer_value, 5.0)
