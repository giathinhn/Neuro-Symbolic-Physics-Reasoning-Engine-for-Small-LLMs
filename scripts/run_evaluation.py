"""Run evaluation benchmark for the full physics reasoning engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_reasoning.core.config import load_config
from physics_reasoning.evaluation.evaluator import Evaluator
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--dataset", default="data/problems/splits/test.jsonl", help="Path to dataset JSONL")
    parser.add_argument("--config", default="configs/experiments/full_system.yaml", help="Path to experiment config")
    parser.add_argument("--output", default="experiments/results", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = PipelineOrchestrator(config=cfg)
    evaluator = Evaluator(pipeline=pipeline, config=cfg)

    problems = evaluator.load_problems(args.dataset)
    evaluator.evaluate_dataset(problems, experiment_name=Path(args.config).stem, output_dir=args.output)


if __name__ == "__main__":
    main()
