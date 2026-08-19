"""Run baseline models (Raw LLM and Calculator) on the benchmark dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_reasoning.core.config import load_config
from physics_reasoning.core.models import ExperimentConfig, ExperimentResult, Problem, Solution
from physics_reasoning.evaluation.baselines import CalculatorBaseline, RawLLMBaseline
from physics_reasoning.evaluation.metrics import compute_all_metrics
from physics_reasoning.llm.provider import LiteLLMProvider


def run_baseline(
    baseline_type: str,
    dataset_path: str,
    model_name: str = "ollama/phi3:mini",
    output_dir: str = "experiments/results",
) -> ExperimentResult:
    """Execute a baseline on the problem dataset."""
    llm = LiteLLMProvider(model_name=model_name)

    if baseline_type == "raw":
        baseline = RawLLMBaseline(llm)
        exp_name = "baseline_raw"
    elif baseline_type == "calculator":
        baseline = CalculatorBaseline(llm)
        exp_name = "baseline_calculator"
    else:
        raise ValueError(f"Unknown baseline: {baseline_type}")

    # Load problems
    problems: list[Problem] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(Problem.model_validate_json(line))

    print(f"Running '{exp_name}' with model '{model_name}' on {len(problems)} problems...")
    start_time = time.perf_counter()
    solutions: list[Solution] = []

    for idx, prob in enumerate(problems, start=1):
        if idx % 10 == 0 or idx == len(problems):
            print(f"  [{idx}/{len(problems)}] Solving {prob.id}...")
        sol = baseline.solve(prob.problem_text, problem_id=prob.id)
        solutions.append(sol)

    duration_s = time.perf_counter() - start_time
    metrics = compute_all_metrics(solutions, problems)

    exp_config = ExperimentConfig(
        experiment_name=exp_name,
        model_name=model_name,
        dataset_split=Path(dataset_path).stem,
        timestamp=datetime.now(),
    )

    result = ExperimentResult(
        config=exp_config,
        metrics=metrics,
        per_problem_results=solutions,
        total_problems=len(problems),
        total_correct=int(metrics.get("answer_accuracy", 0.0) * len(problems)),
        total_duration_s=duration_s,
        total_tokens_used=sum(s.total_tokens for s in solutions),
        total_llm_calls=sum(s.total_llm_calls for s in solutions),
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    print(f"\n{exp_name} completed in {duration_s:.1f}s.")
    print(f"Answer Accuracy: {metrics.get('answer_accuracy', 0.0)*100:.1f}%")
    print(f"Saved results to: {res_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["raw", "calculator"], default="raw")
    parser.add_argument("--dataset", default="data/problems/splits/test.jsonl")
    parser.add_argument("--model", default="ollama/phi3:mini")
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    run_baseline(args.type, args.dataset, model_name=args.model, output_dir=args.output)
