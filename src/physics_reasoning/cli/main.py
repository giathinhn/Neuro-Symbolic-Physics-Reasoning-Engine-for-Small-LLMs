"""Command-line interface for the Neuro-Symbolic Physics Reasoning Engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from physics_reasoning.core.config import load_config
from physics_reasoning.evaluation.evaluator import Evaluator
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Neuro-Symbolic Physics Reasoning Engine CLI."""
    pass


@cli.command()
@click.argument("problem_text", type=str)
@click.option("--model", default=None, help="LLM model name (e.g. 'ollama/phi3:mini', 'gpt-4o-mini')")
@click.option("--config", default=None, help="Path to config YAML file")
@click.option("--json", "output_json", is_flag=True, help="Output solution in JSON format")
@click.option("--verbose", is_flag=True, help="Print detailed solving steps and verification results")
def solve(problem_text: str, model: str | None, config: str | None, output_json: bool, verbose: bool):
    """Solve a physics word problem using the neuro-symbolic engine."""
    cfg = load_config(config)
    if model:
        cfg.model_name = model

    orchestrator = PipelineOrchestrator(config=cfg)
    solution = orchestrator.solve(problem_text)

    if output_json:
        click.echo(solution.model_dump_json(indent=2))
        return

    # Formatted console output
    click.echo("\n========================================================")
    click.echo("  NEURO-SYMBOLIC PHYSICS REASONING ENGINE")
    click.echo("========================================================")
    click.echo(f"Problem: {problem_text}\n")

    if solution.is_verified:
        click.echo(f"[*] Status:   VERIFIED SUCCESS (Attempt #{solution.num_attempts})")
        click.echo(f"[*] Answer:   {solution.answer_value} {solution.answer_unit}")
    else:
        click.echo(f"[!] Status:   UNVERIFIED / FAILED (Attempt #{solution.num_attempts})")
        if solution.answer_value is not None:
            click.echo(f"[!] Answer:   {solution.answer_value} {solution.answer_unit} (Unverified)")
        if solution.error_message:
            click.echo(f"[!] Error:    {solution.error_message}")

    click.echo(f"[*] Formulas: {', '.join(solution.equations_used) if solution.equations_used else 'None'}")
    click.echo(f"[*] Latency:  {solution.latency_ms:.0f} ms | Tokens: {solution.total_tokens}")

    if verbose:
        click.echo("\nExtracted Quantities:")
        for q in solution.quantities_extracted:
            role = "TARGET" if q.is_target else "GIVEN"
            click.echo(f"  - {q.symbol} ({q.name}): {q.value} {q.unit or ''} [{role}]")

        if solution.verification_result:
            click.echo("\nVerification Checks:")
            for check_name in solution.verification_result.checks_performed:
                passed = check_name in solution.verification_result.checks_passed
                mark = "[PASS]" if passed else "[FAIL]"
                click.echo(f"  {mark} {check_name}")

    click.echo("========================================================\n")


@cli.command()
@click.option("--dataset", default="data/problems/splits/test.jsonl", help="Path to evaluation dataset JSONL")
@click.option("--config", default="configs/experiments/full_system.yaml", help="Path to experiment config")
@click.option("--output", default="experiments/results", help="Output directory")
def evaluate(dataset: str, config: str, output: str):
    """Run benchmark evaluation on a problem dataset."""
    cfg = load_config(config)
    orchestrator = PipelineOrchestrator(config=cfg)
    evaluator = Evaluator(pipeline=orchestrator, config=cfg)

    problems = evaluator.load_problems(dataset)
    evaluator.evaluate_dataset(problems, experiment_name=Path(config).stem, output_dir=output)


@cli.command(name="validate-kb")
@click.option("--path", default="data/knowledge", help="Path to knowledge base directory")
def validate_kb_command(path: str):
    """Validate physics knowledge base for syntactic and dimensional consistency."""
    from physics_reasoning.physics.knowledge_base import KnowledgeBase
    from physics_reasoning.solver.expression_parser import parse_equation_string
    from physics_reasoning.units.dimension_checker import DimensionChecker
    from physics_reasoning.units.unit_engine import UnitEngine

    click.echo(f"Validating knowledge base from '{path}'...")
    kb = KnowledgeBase(path)
    try:
        kb.load()
    except Exception as e:
        click.echo(f"Failed to load knowledge base: {e}", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(kb.quantities)} quantities and {len(kb.equations)} equations.")
    dim_checker = DimensionChecker(UnitEngine())

    has_errors = False
    for eq_id, eq in kb.equations.items():
        try:
            parse_equation_string(eq.expression)
        except Exception as e:
            click.echo(f"  [ERROR] Equation '{eq_id}' failed to parse: {e}", err=True)
            has_errors = True
            continue

        var_units = {}
        for var_sym, q_name in eq.variable_quantities.items():
            qty = kb.get_quantity_by_name(q_name)
            if qty and qty.si_unit:
                var_units[var_sym] = qty.si_unit

        res = dim_checker.check_equation(eq.expression, var_units)
        if not res.is_consistent:
            click.echo(f"  [ERROR] {eq_id}: {res.message}", err=True)
            has_errors = True
        else:
            click.echo(f"  [OK] {eq_id:20s}: {eq.expression:35s} [{res.lhs_dimension}]")

    if has_errors:
        click.echo("\nKnowledge base validation FAILED with errors.", err=True)
        sys.exit(1)
    else:
        click.echo("\nKnowledge base validation PASSED successfully!")


@cli.command()
@click.option("--templates", default="data/templates/problem_templates.yaml", help="Path to templates YAML")
@click.option("--count", default=40, help="Number of problems per template")
@click.option("--seed", default=42, help="Random seed")
@click.option("--output", default="data/problems/synthetic/mechanics_synthetic.jsonl", help="Output JSONL path")
def generate(templates: str, count: int, seed: int, output: str):
    """Generate synthetic physics problems from templates."""
    from scripts.generate_synthetic_problems import generate_problems

    generate_problems(templates_path=templates, count_per_template=count, seed=seed, output_path=output)


if __name__ == "__main__":
    cli()
