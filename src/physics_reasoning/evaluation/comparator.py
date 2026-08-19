"""Statistical comparison tools and McNemar's test for benchmark runs."""

from __future__ import annotations

import math
from typing import Any

from physics_reasoning.core.models import ExperimentResult, Problem


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict[str, float]:
    """Perform McNemar's test on paired binary outcome data.

    Tests null hypothesis H0: both systems have equal error rates.

    Contingency table:
                System B Correct   System B Incorrect
    System A Correct        n11                n10 (b)
    System A Incorrect      n01 (c)            n00

    chi-squared = (|b - c| - 1)^2 / (b + c) (with Yates' continuity correction)
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("Paired data arrays must have identical length")

    # b: A correct, B incorrect
    b = sum(1 for a, b_val in zip(correct_a, correct_b) if a and not b_val)
    # c: A incorrect, B correct
    c = sum(1 for a, b_val in zip(correct_a, correct_b) if not a and b_val)

    if (b + c) == 0:
        return {"chi2": 0.0, "p_value": 1.0, "is_significant": False, "b": b, "c": c}

    # Chi-square statistic with continuity correction
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)

    # Approximate 1-df chi-square p-value using erfc
    # p ≈ erfc(sqrt(chi2 / 2))
    p_value = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "is_significant": bool(p_value < 0.05),
        "b_favors_a": b,
        "c_favors_b": c,
    }


def compare_experiments(
    results: dict[str, ExperimentResult], problems: list[Problem]
) -> dict[str, Any]:
    """Compare multiple experiment runs and perform pairwise significance tests."""
    prob_map = {p.id: p for p in problems}

    comparison: dict[str, Any] = {
        "summary": {},
        "pairwise_significance": {},
    }

    # Extract metrics summary for each experiment
    for name, exp_res in results.items():
        comparison["summary"][name] = exp_res.metrics

    # Pairwise McNemar's test for accuracy
    exp_names = list(results.keys())
    for i in range(len(exp_names)):
        for j in range(i + 1, len(exp_names)):
            name_a = exp_names[i]
            name_b = exp_names[j]

            res_a = results[name_a]
            res_b = results[name_b]

            sol_map_a = {s.problem_id: s for s in res_a.per_problem_results}
            sol_map_b = {s.problem_id: s for s in res_b.per_problem_results}

            correct_a: list[bool] = []
            correct_b: list[bool] = []

            for p in problems:
                actual = p.answer_value

                sol_a = sol_map_a.get(p.id)
                a_ok = bool(
                    sol_a
                    and sol_a.answer_value is not None
                    and math.isclose(sol_a.answer_value, actual, rel_tol=0.01, abs_tol=1e-5)
                )

                sol_b = sol_map_b.get(p.id)
                b_ok = bool(
                    sol_b
                    and sol_b.answer_value is not None
                    and math.isclose(sol_b.answer_value, actual, rel_tol=0.01, abs_tol=1e-5)
                )

                correct_a.append(a_ok)
                correct_b.append(b_ok)

            test_res = mcnemar_test(correct_a, correct_b)
            pair_key = f"{name_a}_vs_{name_b}"
            comparison["pairwise_significance"][pair_key] = test_res

    return comparison
