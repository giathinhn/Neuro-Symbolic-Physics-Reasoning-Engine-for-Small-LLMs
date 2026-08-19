"""Prompt templates for system, solve, repair, and tool calling modes."""

from __future__ import annotations

import json
from typing import Any

from physics_reasoning.core.models import (
    LLMParsedOutput,
    VerificationError,
)

SYSTEM_PROMPT_TEMPLATE = """You are an expert physics problem analyst. Your task is to analyze physics word problems and extract structured physical concepts, quantities, and equations.

CRITICAL INSTRUCTIONS:
1. You do NOT need to perform manual arithmetic calculations. The deterministic symbolic solver (SymPy) will compute the mathematical solution.
2. Identify all GIVEN quantities, including implicit ones (e.g., "starts from rest" implies initial_velocity v_i = 0 m/s; "dropped under gravity" implies acceleration a = 9.8 m/s^2).
3. Identify the TARGET quantity to solve for.
4. Select the standard physical equation(s) that relate the known quantities to the target.
5. Use standard physical variable symbols (F, m, a, v, d, t, v_i, v_f, KE, PE, W_work, P, etc.).
6. Always return a valid JSON object matching the requested schema.

{available_equations_section}
"""

JSON_SCHEMA_EXAMPLE = """{
  "problem_understanding": "Brief restatement of the problem",
  "quantities": [
    {"name": "mass", "symbol": "m", "value": 2.0, "unit": "kg", "role": "given"},
    {"name": "force", "symbol": "F", "value": 10.0, "unit": "N", "role": "given"},
    {"name": "acceleration", "symbol": "a", "value": null, "unit": "m/s**2", "role": "target"}
  ],
  "equations": [
    {"equation_id": "newton2", "expression": "F = m * a", "justification": "Newton's second law relates net force, mass, and acceleration."}
  ],
  "target_variable": "a",
  "solution_steps": [
    "Apply Newton's second law: F = m * a",
    "Substitute known values: F = 10 N, m = 2 kg",
    "Solve for a: a = F / m"
  ],
  "proposed_answer": null,
  "proposed_unit": "m/s**2"
}"""


def build_system_prompt(available_equations: list[str] | None = None) -> str:
    """Build the system prompt for the LLM."""
    if available_equations:
        eq_list = "\n".join(f"- {eq}" for eq in available_equations[:25])
        section = f"Standard physics equations available:\n{eq_list}"
    else:
        section = ""
    return SYSTEM_PROMPT_TEMPLATE.format(available_equations_section=section)


def build_solve_prompt(problem_text: str) -> str:
    """Build the user prompt for analyzing a physics problem."""
    return f"""Please analyze this physics problem and extract the structured quantities and equations:

PROBLEM:
"{problem_text}"

Respond ONLY with a JSON object following this format:
```json
{JSON_SCHEMA_EXAMPLE}
```
"""


def build_repair_prompt(
    previous_output: LLMParsedOutput | str,
    verification_errors: list[VerificationError],
    attempt_number: int,
) -> str:
    """Build a repair prompt when verification checks fail."""
    error_lines: list[str] = []
    for i, err in enumerate(verification_errors, start=1):
        line = f"{i}. [{err.error_type.value.upper()}] ({err.severity.value.upper()}): {err.message}"
        if err.suggestion:
            line += f"\n   Suggestion: {err.suggestion}"
        error_lines.append(line)

    errors_formatted = "\n".join(error_lines)

    prev_text = (
        previous_output.model_dump_json(indent=2)
        if isinstance(previous_output, LLMParsedOutput)
        else str(previous_output)
    )

    return f"""The previous solution attempt (Attempt #{attempt_number}) failed verification checks with the following issues:

{errors_formatted}

PREVIOUS OUTPUT:
```json
{prev_text}
```

Please carefully correct the identified issues and output a revised JSON solution.
Ensure all equations are dimensionally valid and all units match standard physics.
"""


def build_tool_call_prompt(problem_text: str) -> str:
    """Build a prompt for tool-calling mode."""
    return f"""Solve the following physics problem using the available tools:

PROBLEM:
"{problem_text}"

Available tools:
- search_equations: Find relevant equations from knowledge base
- solve_equation: Deterministically solve equations for target variable
- convert_units: Convert units to SI
- check_dimensions: Check dimensional consistency
- verify_solution: Verify answer by back-substitution
- calculate: Evaluate arithmetic expressions

Use tools step-by-step to arrive at a verified solution.
"""
