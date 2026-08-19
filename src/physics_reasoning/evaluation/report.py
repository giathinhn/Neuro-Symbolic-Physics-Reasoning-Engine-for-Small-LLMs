"""Report generation for experiment results and comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physics_reasoning.core.models import ExperimentResult


def generate_markdown_report(
    experiment_results: dict[str, ExperimentResult],
    comparison_data: dict[str, Any] | None = None,
) -> str:
    """Generate a clean GitHub-flavored markdown report of evaluation results."""
    lines: list[str] = []

    lines.append("# Neuro-Symbolic Physics Reasoning Evaluation Report\n")
    lines.append("## 1. Overall Performance Comparison\n")

    # Table header
    lines.append(
        "| Experiment / Baseline | Answer Accuracy | Exact Match | Equation Acc | Unit Acc | Hallucination Rate | Invalid Eq Rate | Mean Latency (ms) | Verified Rate |"
    )
    lines.append(
        "|:----------------------|:---------------:|:-----------:|:------------:|:--------:|:------------------:|:---------------:|:-----------------:|:-------------:|"
    )

    for name, res in experiment_results.items():
        m = res.metrics
        acc = f"{m.get('answer_accuracy', 0.0) * 100:.1f}%"
        em = f"{m.get('exact_match', 0.0) * 100:.1f}%"
        eq_acc = f"{m.get('equation_accuracy', 0.0) * 100:.1f}%"
        unit_acc = f"{m.get('unit_accuracy', 0.0) * 100:.1f}%"
        halluc = f"{m.get('hallucination_rate', 0.0) * 100:.1f}%"
        inv_eq = f"{m.get('invalid_equation_rate', 0.0) * 100:.1f}%"
        latency = f"{m.get('mean_latency_ms', 0.0):.0f}"
        verified = f"{m.get('verified_rate', 0.0) * 100:.1f}%"

        lines.append(
            f"| **{name}** | {acc} | {em} | {eq_acc} | {unit_acc} | {halluc} | {inv_eq} | {latency} | {verified} |"
        )

    lines.append("\n## 2. Statistical Significance (McNemar's Test, p < 0.05)\n")

    if comparison_data and "pairwise_significance" in comparison_data:
        sig_data = comparison_data["pairwise_significance"]
        lines.append("| Comparison Pair | Chi-Square | p-value | Statistically Significant? |")
        lines.append("|:----------------|:----------:|:-------:|:--------------------------:|")

        for pair_name, test_res in sig_data.items():
            chi2 = f"{test_res.get('chi2', 0.0):.3f}"
            pval = f"{test_res.get('p_value', 1.0):.4e}"
            is_sig = "**YES** (p < 0.05)" if test_res.get("is_significant") else "No"
            lines.append(f"| `{pair_name}` | {chi2} | {pval} | {is_sig} |")
    else:
        lines.append("*No pairwise significance comparisons computed.*\n")

    return "\n".join(lines)


def save_report(
    report_text: str, output_path: str = "experiments/results/evaluation_report.md"
) -> None:
    """Save markdown report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
