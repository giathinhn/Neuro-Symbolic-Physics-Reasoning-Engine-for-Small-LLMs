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


# ---------------------------------------------------------------------------
# Qualitative Phenomenon Explanation Prompts
# ---------------------------------------------------------------------------

QUALITATIVE_SYSTEM_PROMPT_TEMPLATE = """You are an expert physics educator and reasoning engine specializing in Middle School (THCS) Physics Phenomenon Explanations.
Your task is to explain physics phenomena using rigorous, step-by-step physical principles and causal reasoning chains.

CRITICAL REASONING RULES:
1. Identify the core physical principle/law (e.g. Quán tính, Áp suất p = F/S, Lực ma sát, Lực đẩy Ác-si-mét, Sự dãn nở vì nhiệt không đều, Dẫn nhiệt, Sự bay hơi thu nhiệt...).
2. Form a complete, unbroken Causal Chain:
   - Step 1: Initial State & External Action (Trạng thái ban đầu và tác động/thay đổi xảy ra).
   - Step 2: Physical Mechanism (Cơ chế vật lý theo định luật: tại sao các bộ phận phản ứng khác nhau do tính chất vật lý).
   - Step 3: Observed Consequence (Hệ quả cuối cùng giải thích đúng hiện tượng được hỏi).
3. AVOID COMMON MISCONCEPTIONS:
   - NEVER refer to "inertia" as a force (quán tính không phải là lực đẩy/kéo).
   - Do NOT confuse force (áp lực F) with pressure (áp suất p = F/S).
   - Do NOT confuse heat (nhiệt lượng) with temperature (nhiệt độ).
   - Do NOT claim objects sink solely because they are heavy (chìm/nổi do trọng lượng riêng so với chất lỏng).
4. Always respond with a structured JSON object matching the requested schema.

{available_principles_section}
"""

QUALITATIVE_JSON_SCHEMA_EXAMPLE = """{
  "problem_understanding": "Tóm tắt hiện tượng cần giải thích",
  "observed_phenomenon": "Hiện tượng được quan sát trong thực tế",
  "core_principles": ["inertia_law"],
  "causal_chain": [
    {
      "step_number": 1,
      "state_or_action": "Xe và hành khách đang cùng chuyển động về phía trước với vận tốc v.",
      "physical_mechanism": "Khi xe phanh gấp, lực ma sát giữa bánh xe và mặt đường làm xe và phần dưới cơ thể hành khách tiếp xúc với ghế/sàn xe giảm nhanh vận tốc.",
      "governing_principle": "inertia_law"
    },
    {
      "step_number": 2,
      "state_or_action": "Phần thân trên của hành khách chưa chịu lực hãm ngay.",
      "physical_mechanism": "Do có quán tính, phần thân trên tiếp tục duy trì vận tốc v ban đầu hướng về phía trước.",
      "governing_principle": "inertia_law"
    },
    {
      "step_number": 3,
      "state_or_action": "Kết quả tổng thể của chuyển động",
      "physical_mechanism": "Phần thân trên di chuyển nhanh hơn phần thân dưới, khiến hành khách có xu hướng bị ngã chúi về phía trước.",
      "governing_principle": "inertia_law"
    }
  ],
  "conclusion": "Khi xe phanh gấp, do có quán tính nên phần thân trên của hành khách tiếp tục duy trì vận tốc cũ trong khi phần dưới đã dừng lại cùng xe, khiến người bị ngã chúi về phía trước.",
  "scientific_keywords": ["quán tính", "duy trì vận tốc", "phanh gấp", "chuyển động"]
}"""


def build_qualitative_system_prompt(available_principles: list[str] | None = None) -> str:
    """Build the system prompt for qualitative physics phenomenon explanations."""
    if available_principles:
        p_list = "\n".join(f"- {p}" for p in available_principles[:15])
        section = f"Standard qualitative physics principles available in knowledge base:\n{p_list}"
    else:
        section = ""
    return QUALITATIVE_SYSTEM_PROMPT_TEMPLATE.format(available_principles_section=section)


def build_qualitative_solve_prompt(problem_text: str) -> str:
    """Build user prompt for explaining a qualitative physics phenomenon."""
    return f"""Please provide a rigorous, scientifically grounded explanation for the following physics phenomenon:

PHENOMENON QUESTION:
"{problem_text}"

Respond ONLY with a JSON object following this format:
```json
{QUALITATIVE_JSON_SCHEMA_EXAMPLE}
```
"""


def build_qualitative_repair_prompt(
    previous_output: str,
    verification_errors: list[VerificationError],
    attempt_number: int,
) -> str:
    """Build repair prompt for fixing qualitative explanation errors and misconceptions."""
    error_lines: list[str] = []
    for i, err in enumerate(verification_errors, start=1):
        line = f"{i}. [{err.error_type.value.upper()}] ({err.severity.value.upper()}): {err.message}"
        if err.suggestion:
            line += f"\n   Gợi ý khắc phục: {err.suggestion}"
        error_lines.append(line)

    errors_formatted = "\n".join(error_lines)

    return f"""Lời giải thích hiện tượng ở lần thử trước (Lần #{attempt_number}) chưa đạt yêu cầu hoặc mắc ngộ nhận vật lý sau:

{errors_formatted}

LỜI GIẢI CŨ:
```json
{previous_output}
```

Vui lòng chỉnh sửa lại lời giải thích chặt chẽ, loại bỏ hoàn toàn các ngộ nhận vật lý trên và trả về định dạng JSON chuẩn.
"""
