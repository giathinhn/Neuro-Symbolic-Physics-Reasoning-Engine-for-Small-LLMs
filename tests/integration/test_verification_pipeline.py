"""Integration tests for verification pipeline."""

from __future__ import annotations

import pytest

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.core.enums import ErrorType, QuantityRole
from physics_reasoning.core.models import (
    LLMParsedOutput,
    ParsedEquation,
    ParsedQuantity,
    SolveResult,
)
from physics_reasoning.verifier.verification_pipeline import VerificationPipeline


@pytest.fixture
def verification_pipeline() -> VerificationPipeline:
    config = PipelineConfig()
    return VerificationPipeline(config=config)


class TestVerificationPipelineIntegration:
    def test_full_verification_pass(self, verification_pipeline):
        parsed = LLMParsedOutput(
            problem_understanding="Find acceleration",
            quantities=[
                ParsedQuantity(name="force", symbol="F", value=10.0, unit="N", role=QuantityRole.GIVEN),
                ParsedQuantity(name="mass", symbol="m", value=2.0, unit="kg", role=QuantityRole.GIVEN),
                ParsedQuantity(name="acceleration", symbol="a", unit="m/s**2", role=QuantityRole.TARGET),
            ],
            equations=[ParsedEquation(expression="F = m * a")],
            target_variable="a",
            proposed_unit="m/s**2",
        )
        solve_res = SolveResult(
            target_variable="a",
            solutions=[5.0],
            is_numeric=True,
        )

        res = verification_pipeline.verify(parsed, solve_res, [])
        assert res.is_valid
        assert len(res.errors) == 0
        assert res.confidence == 1.0

    def test_full_verification_fail_dimension_mismatch(self, verification_pipeline):
        parsed = LLMParsedOutput(
            problem_understanding="Wrong equation test",
            quantities=[
                ParsedQuantity(name="force", symbol="F", value=10.0, unit="N", role=QuantityRole.GIVEN),
                ParsedQuantity(name="mass", symbol="m", value=2.0, unit="kg", role=QuantityRole.GIVEN),
                ParsedQuantity(name="time", symbol="t", value=5.0, unit="s", role=QuantityRole.GIVEN),
            ],
            equations=[ParsedEquation(expression="F = m * t")],
            target_variable="F",
        )
        solve_res = SolveResult(
            target_variable="F",
            solutions=[10.0],
            is_numeric=True,
        )

        res = verification_pipeline.verify(parsed, solve_res, [])
        assert not res.is_valid
        assert any(e.error_type == ErrorType.DIMENSION_MISMATCH for e in res.errors)
        assert res.confidence == 0.0
