"""Generate synthetic physics word problems from templates with deterministic ground truth."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_reasoning.core.enums import Difficulty, ProblemSource
from physics_reasoning.core.models import PhysicsQuantity, Problem
from physics_reasoning.physics.constants import PHYSICAL_CONSTANTS
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.solver.numerical import round_to_significant_figures
from physics_reasoning.solver.symbolic_solver import SymbolicSolver
from physics_reasoning.units.unit_engine import UnitEngine


def generate_problems(
    templates_path: str = "data/templates/problem_templates.yaml",
    count_per_template: int = 40,
    seed: int = 42,
    output_path: str = "data/problems/synthetic/mechanics_synthetic.jsonl",
) -> list[Problem]:
    """Generate synthetic physics problems from templates."""
    random.seed(seed)

    t_file = Path(templates_path)
    if not t_file.exists():
        raise FileNotFoundError(f"Template file not found: {templates_path}")

    with open(t_file, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    templates = data.get("templates", [])
    print(f"Loaded {len(templates)} templates from {templates_path}")

    kb = KnowledgeBase()
    kb.load()
    solver = SymbolicSolver()
    unit_engine = UnitEngine()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    generated_problems: list[Problem] = []
    problem_counter = 0

    for t in templates:
        t_id = t["id"]
        topic = t["topic"]
        difficulty = Difficulty(t.get("difficulty", "easy"))
        template_str = t["template"]
        req_eqs = t["required_equations"]
        target_name = t["target"]
        target_symbol = t["target_symbol"]
        target_unit = t["target_unit"]
        given_vars = t.get("given_variables", [])

        # Get KB equation
        kb_eq = kb.get_equation(req_eqs[0])
        eq_expr = kb_eq.expression if kb_eq else req_eqs[0]

        for i in range(count_per_template):
            problem_counter += 1
            pid = f"syn_{t_id}_{i+1:04d}"

            # Sample values & units for each given variable
            format_dict: dict[str, Any] = {}
            given_quantities: list[PhysicsQuantity] = []
            known_values_si: dict[str, float] = dict(PHYSICAL_CONSTANTS)

            for g in given_vars:
                sym = g["symbol"]
                name = g["name"]
                val_range = g["value_range"]
                unit_choices = [u for u in g["unit_choices"] if u] or [""]

                # Sample numeric value
                min_v, max_v = val_range[0], val_range[1]
                if min_v == max_v:
                    val = min_v
                elif isinstance(min_v, int) and isinstance(max_v, int):
                    val = float(random.randint(min_v, max_v))
                else:
                    val = round(random.uniform(min_v, max_v), 2)

                unit = random.choice(unit_choices) if unit_choices[0] else ""

                # Populate template string args
                format_dict[f"{sym}_val"] = int(val) if val == int(val) else val
                format_dict[f"{sym}_unit"] = unit

                # Convert to SI for solving
                if unit:
                    try:
                        si_val, _ = unit_engine.to_si(val, unit)
                    except Exception:
                        si_val = val
                else:
                    si_val = val

                known_values_si[sym] = si_val

                given_quantities.append(
                    PhysicsQuantity(
                        name=name,
                        symbol=sym,
                        value=val,
                        unit=unit or None,
                        is_given=True,
                    )
                )

            # Format the natural language problem text
            try:
                problem_text = template_str.format(**format_dict)
            except KeyError as e:
                print(f"Skipping template {t_id} due to format key error: {e}")
                continue

            # Deterministically solve for ground truth answer using SymPy
            solve_res = solver.solve_single(eq_expr, known_values_si, target_symbol)
            if not solve_res.is_numeric or not solve_res.solutions:
                continue

            raw_sol = solve_res.solutions[0]
            # Convert answer to target unit if needed
            answer_val = float(raw_sol)
            answer_val = round_to_significant_figures(answer_val, sig_figs=4)

            target_quantity = PhysicsQuantity(
                name=target_name,
                symbol=target_symbol,
                unit=target_unit,
                is_target=True,
            )

            problem = Problem(
                id=pid,
                problem_text=problem_text,
                topic=topic,
                difficulty=difficulty,
                source=ProblemSource.SYNTHETIC,
                given_quantities=given_quantities,
                target_quantity=target_quantity,
                required_equations=req_eqs,
                answer_value=answer_val,
                answer_unit=target_unit,
                reasoning_steps=[
                    f"Use equation: {eq_expr}",
                    f"Substitute known values to solve for {target_symbol}",
                    f"{target_symbol} = {answer_val} {target_unit}",
                ],
            )
            generated_problems.append(problem)

    print(f"Generated {len(generated_problems)} valid synthetic problems.")

    # Write out JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for p in generated_problems:
            f.write(p.model_dump_json() + "\n")

    print(f"Saved problems to {output_path}")
    return generated_problems


if __name__ == "__main__":
    generate_problems()
