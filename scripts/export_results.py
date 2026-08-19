"""Export experiment results and generate Markdown summary reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_reasoning.core.models import ExperimentResult, Problem
from physics_reasoning.evaluation.comparator import compare_experiments
from physics_reasoning.evaluation.report import generate_markdown_report, save_report


def export_reports(
    results_dir: str = "experiments/results",
    dataset_path: str = "data/problems/splits/test.jsonl",
    output_report_path: str = "experiments/results/evaluation_report.md",
) -> None:
    """Collect all result JSONs, run statistical comparisons, and output report."""
    res_dir = Path(results_dir)
    json_files = list(res_dir.glob("*.json"))

    if not json_files:
        print(f"No experiment result JSON files found in {results_dir}")
        return

    # Load problems
    problems: list[Problem] = []
    if Path(dataset_path).exists():
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problems.append(Problem.model_validate_json(line))

    # Load experiment results
    results_map: dict[str, ExperimentResult] = {}
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
                res = ExperimentResult.model_validate(data)
                results_map[res.config.experiment_name] = res
        except Exception as e:
            print(f"Skipping {jf}: {e}")

    print(f"Loaded {len(results_map)} experiment runs: {list(results_map.keys())}")

    comparison = None
    if problems and len(results_map) > 1:
        comparison = compare_experiments(results_map, problems)

    report_md = generate_markdown_report(results_map, comparison)
    save_report(report_md, output_report_path)

    print(f"\nMarkdown report exported to {output_report_path}")
    print("\n" + report_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--dataset", default="data/problems/splits/test.jsonl")
    parser.add_argument("--output", default="experiments/results/evaluation_report.md")
    args = parser.parse_args()

    export_reports(args.results_dir, args.dataset, args.output)
