"""Tests for evaluation metrics and statistical comparator."""

from __future__ import annotations

import pytest

from physics_reasoning.core.enums import Difficulty, ErrorSeverity, ErrorType, ProblemSource
from physics_reasoning.core.models import (
    Problem,
    Solution,
    VerificationError,
    VerificationResult,
)
from physics_reasoning.evaluation.comparator import mcnemar_test
from physics_reasoning.evaluation.metrics import (
    compute_all_metrics,
    compute_answer_accuracy,
    compute_exact_match,
    compute_hallucination_rate,
    compute_invalid_equation_rate,
    compute_unit_accuracy,
)


@pytest.fixture
def sample_problems() -> list[Problem]:
    return [
        Problem(
            id="p1",
            problem_text="Find acceleration",
            topic="newton_laws",
            difficulty=Difficulty.EASY,
            source=ProblemSource.MANUAL,
            required_equations=["newton2"],
            answer_value=5.0,
            answer_unit="m/s**2",
        ),
        Problem(
            id="p2",
            problem_text="Find velocity",
            topic="kinematics",
            difficulty=Difficulty.EASY,
            source=ProblemSource.MANUAL,
            required_equations=["kin_vel_def"],
            answer_value=20.0,
            answer_unit="m/s",
        ),
        Problem(
            id="p3",
            problem_text="Find force",
            topic="newton_laws",
            difficulty=Difficulty.EASY,
            source=ProblemSource.MANUAL,
            required_equations=["newton2"],
            answer_value=100.0,
            answer_unit="N",
        ),
    ]


@pytest.fixture
def sample_solutions() -> list[Solution]:
    return [
        Solution(
            problem_id="p1",
            answer_value=5.001,  # within 1% tol
            answer_unit="m/s**2",
            equations_used=["newton2"],
            is_verified=True,
            num_attempts=1,
            total_tokens=100,
            latency_ms=500.0,
        ),
        Solution(
            problem_id="p2",
            answer_value=15.0,  # incorrect answer
            answer_unit="m/s",
            equations_used=["kin_vel_def"],
            is_verified=True,
            num_attempts=2,
            total_tokens=200,
            latency_ms=800.0,
        ),
        Solution(
            problem_id="p3",
            answer_value=100.0,  # exact
            answer_unit="N",
            equations_used=["newton2"],
            is_verified=True,
            num_attempts=1,
            total_tokens=120,
            latency_ms=450.0,
        ),
    ]


class TestMetrics:
    def test_compute_answer_accuracy(self, sample_solutions, sample_problems):
        # 2 out of 3 correct
        acc = compute_answer_accuracy(sample_solutions, sample_problems)
        assert pytest.approx(acc, 0.01) == 2 / 3

    def test_compute_exact_match(self, sample_solutions, sample_problems):
        # p1 (5.001 -> 5.0) and p3 (100.0) match to 2 decimals -> 2 out of 3
        em = compute_exact_match(sample_solutions, sample_problems)
        assert pytest.approx(em, 0.01) == 2 / 3

    def test_compute_unit_accuracy(self, sample_solutions, sample_problems):
        # all 3 units match
        ua = compute_unit_accuracy(sample_solutions, sample_problems)
        assert pytest.approx(ua, 0.01) == 1.0

    def test_compute_hallucination_rate(self):
        sols = [
            Solution(
                problem_id="p1",
                verification_result=VerificationResult(
                    is_valid=False,
                    errors=[
                        VerificationError(
                            error_type=ErrorType.INVALID_EQUATION,
                            severity=ErrorSeverity.ERROR,
                            message="Invalid",
                        )
                    ],
                ),
            ),
            Solution(problem_id="p2", verification_result=VerificationResult(is_valid=True)),
        ]
        assert compute_hallucination_rate(sols) == 0.5

    def test_mcnemar_test_identical(self):
        a = [True, True, False, True]
        b = [True, True, False, True]
        res = mcnemar_test(a, b)
        assert res["p_value"] == 1.0
        assert not res["is_significant"]

    def test_mcnemar_test_different(self):
        a = [True] * 50 + [False] * 50
        b = [False] * 50 + [False] * 50  # B is always wrong
        res = mcnemar_test(a, b)
        assert res["is_significant"]
        assert res["p_value"] < 0.001
