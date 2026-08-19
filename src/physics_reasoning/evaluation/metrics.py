"""Evaluation metrics computation for accuracy, reliability, and efficiency."""

from __future__ import annotations

import math
import statistics
from typing import Any

from physics_reasoning.core.enums import ErrorType
from physics_reasoning.core.models import Problem, Solution
from physics_reasoning.units.unit_engine import UnitEngine


def compute_answer_accuracy(
    solutions: list[Solution], problems: list[Problem], rel_tol: float = 0.01
) -> float:
    """Fraction of problems with numerically correct answer (|pred - actual| / |actual| < rel_tol)."""
    if not problems:
        return 0.0

    prob_map = {p.id: p for p in problems}
    correct = 0

    for sol in solutions:
        prob = prob_map.get(sol.problem_id)
        if not prob or sol.answer_value is None:
            continue

        actual = prob.answer_value
        pred = sol.answer_value

        if not math.isfinite(pred):
            continue

        # Check closeness
        if math.isclose(pred, actual, rel_tol=rel_tol, abs_tol=1e-5):
            correct += 1

    return correct / len(problems)


def compute_exact_match(
    solutions: list[Solution], problems: list[Problem]
) -> float:
    """Fraction of problems matching actual answer to 2 decimal places."""
    if not problems:
        return 0.0

    prob_map = {p.id: p for p in problems}
    correct = 0

    for sol in solutions:
        prob = prob_map.get(sol.problem_id)
        if not prob or sol.answer_value is None:
            continue

        if round(sol.answer_value, 2) == round(prob.answer_value, 2):
            correct += 1

    return correct / len(problems)


def compute_equation_accuracy(
    solutions: list[Solution], problems: list[Problem]
) -> float:
    """Fraction of problems where at least one required equation was correctly identified."""
    if not problems:
        return 0.0

    prob_map = {p.id: p for p in problems}
    correct = 0

    for sol in solutions:
        prob = prob_map.get(sol.problem_id)
        if not prob:
            continue

        used_set = set(sol.equations_used)
        req_set = set(prob.required_equations)

        if len(used_set.intersection(req_set)) > 0:
            correct += 1

    return correct / len(problems)


def compute_unit_accuracy(
    solutions: list[Solution], problems: list[Problem], unit_engine: UnitEngine | None = None
) -> float:
    """Fraction of problems where the answer unit is dimensionally compatible with ground truth."""
    if not problems:
        return 0.0

    ue = unit_engine or UnitEngine()
    prob_map = {p.id: p for p in problems}
    correct = 0

    for sol in solutions:
        prob = prob_map.get(sol.problem_id)
        if not prob or not sol.answer_unit:
            continue

        if ue.are_compatible(sol.answer_unit, prob.answer_unit):
            correct += 1

    return correct / len(problems)


def compute_hallucination_rate(solutions: list[Solution]) -> float:
    """Fraction of problems where model proposed an invalid/hallucinated equation."""
    if not solutions:
        return 0.0

    count = 0
    for sol in solutions:
        if sol.verification_result and any(
            e.error_type == ErrorType.INVALID_EQUATION
            for e in sol.verification_result.errors
        ):
            count += 1

    return count / len(solutions)


def compute_invalid_equation_rate(solutions: list[Solution]) -> float:
    """Fraction of problems where dimensional inconsistency was detected."""
    if not solutions:
        return 0.0

    count = 0
    for sol in solutions:
        if sol.verification_result and any(
            e.error_type == ErrorType.DIMENSION_MISMATCH
            for e in sol.verification_result.errors
        ):
            count += 1

    return count / len(solutions)


def compute_correction_success_rate(solutions: list[Solution]) -> float:
    """Of problems that required retries, fraction that successfully became verified."""
    retried = [s for s in solutions if s.num_attempts > 1]
    if not retried:
        return 0.0

    success = [s for s in retried if s.is_verified]
    return len(success) / len(retried)


def compute_efficiency_metrics(solutions: list[Solution]) -> dict[str, float]:
    """Compute latency, tokens, and attempt efficiency metrics."""
    if not solutions:
        return {
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "mean_tokens": 0.0,
            "mean_llm_calls": 0.0,
            "verified_rate": 0.0,
        }

    latencies = [s.latency_ms for s in solutions]
    tokens = [s.total_tokens for s in solutions]
    llm_calls = [s.total_llm_calls for s in solutions]
    verified_count = sum(1 for s in solutions if s.is_verified)

    return {
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "mean_tokens": statistics.mean(tokens),
        "mean_llm_calls": statistics.mean(llm_calls),
        "verified_rate": verified_count / len(solutions),
    }


def compute_all_metrics(
    solutions: list[Solution], problems: list[Problem], unit_engine: UnitEngine | None = None
) -> dict[str, float]:
    """Compute complete metric dictionary for an experiment run."""
    metrics: dict[str, float] = {}

    metrics["answer_accuracy"] = compute_answer_accuracy(solutions, problems)
    metrics["exact_match"] = compute_exact_match(solutions, problems)
    metrics["equation_accuracy"] = compute_equation_accuracy(solutions, problems)
    metrics["unit_accuracy"] = compute_unit_accuracy(solutions, problems, unit_engine=unit_engine)
    metrics["hallucination_rate"] = compute_hallucination_rate(solutions)
    metrics["invalid_equation_rate"] = compute_invalid_equation_rate(solutions)
    metrics["correction_success_rate"] = compute_correction_success_rate(solutions)

    efficiency = compute_efficiency_metrics(solutions)
    metrics.update(efficiency)

    return metrics
