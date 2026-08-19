"""Evaluator runner executing experiments across benchmark splits."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.core.models import (
    ExperimentConfig,
    ExperimentResult,
    Problem,
    Solution,
)
from physics_reasoning.evaluation.metrics import compute_all_metrics
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator


class Evaluator:
    """Benchmark runner executing problems on the pipeline."""

    def __init__(
        self,
        pipeline: PipelineOrchestrator,
        config: PipelineConfig | None = None,
    ):
        self.pipeline = pipeline
        self.config = config or pipeline.config

    def load_problems(self, dataset_path: str | Path) -> list[Problem]:
        """Load problems from a JSONL file."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        problems: list[Problem] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    problems.append(Problem.model_validate(data))

        return problems

    def evaluate_dataset(
        self,
        problems: list[Problem],
        experiment_name: str = "full_system",
        output_dir: str = "experiments/results",
    ) -> ExperimentResult:
        """Run benchmark evaluation on a list of problems."""
        start_time = time.perf_counter()
        solutions: list[Solution] = []

        print(f"\nStarting evaluation '{experiment_name}' on {len(problems)} problems...")

        for idx, prob in enumerate(problems, start=1):
            if idx % 10 == 0 or idx == len(problems):
                print(f"  [{idx}/{len(problems)}] Evaluating problem {prob.id}...")

            try:
                sol = self.pipeline.solve(prob.problem_text, problem_id=prob.id)
                solutions.append(sol)
            except Exception as e:
                # Capture unhandled error into solution
                failed_sol = Solution(
                    problem_id=prob.id,
                    is_verified=False,
                    error_message=f"Evaluation execution error: {e}",
                )
                solutions.append(failed_sol)

        total_duration_s = time.perf_counter() - start_time
        metrics = compute_all_metrics(solutions, problems, unit_engine=self.pipeline.unit_engine)

        total_correct = int(metrics.get("answer_accuracy", 0.0) * len(problems))
        total_tokens = sum(s.total_tokens for s in solutions)
        total_llm_calls = sum(s.total_llm_calls for s in solutions)

        exp_config = ExperimentConfig(
            experiment_name=experiment_name,
            model_name=self.config.model_name,
            system_config=self.config.model_dump(),
            dataset_split="custom",
            timestamp=datetime.now(),
            random_seed=self.config.random_seed,
            max_retries=self.config.max_retries,
        )

        result = ExperimentResult(
            config=exp_config,
            metrics=metrics,
            per_problem_results=solutions,
            total_problems=len(problems),
            total_correct=total_correct,
            total_duration_s=total_duration_s,
            total_tokens_used=total_tokens,
            total_llm_calls=total_llm_calls,
        )

        # Save result JSON
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_file = out_dir / f"{experiment_name}_{timestamp_str}.json"

        with open(res_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        print(f"\nEvaluation finished in {total_duration_s:.1f}s.")
        print(f"Answer Accuracy: {metrics.get('answer_accuracy', 0.0) * 100:.1f}%")
        print(f"Results saved to: {res_file}")

        return result
