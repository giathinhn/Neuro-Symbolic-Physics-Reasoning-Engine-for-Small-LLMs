"""Run ablation studies over the physics reasoning engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_reasoning.core.config import PipelineConfig, load_config
from physics_reasoning.evaluation.evaluator import Evaluator
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/problems/splits/dev.jsonl")
    parser.add_argument("--ablations-config", default="configs/experiments/ablations.yaml")
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    with open(args.ablations_config, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    ablations = data.get("ablations", [])
    print(f"Loaded {len(ablations)} ablation conditions.")

    base_config = load_config()

    for ab in ablations:
        ab_id = ab["id"]
        ab_name = ab.get("name", ab_id)
        print(f"\n========================================================")
        print(f"Running ablation: {ab_name} ({ab_id})")
        print(f"========================================================")

        # Merge base config with ablation overrides
        cfg_dict = base_config.model_dump()
        for k, v in ab.items():
            if k not in ("id", "name") and k in cfg_dict:
                cfg_dict[k] = v

        ab_cfg = PipelineConfig(**cfg_dict)
        pipeline = PipelineOrchestrator(config=ab_cfg)
        evaluator = Evaluator(pipeline=pipeline, config=ab_cfg)

        problems = evaluator.load_problems(args.dataset)
        evaluator.evaluate_dataset(problems, experiment_name=f"ablation_{ab_id}", output_dir=args.output)


if __name__ == "__main__":
    main()
