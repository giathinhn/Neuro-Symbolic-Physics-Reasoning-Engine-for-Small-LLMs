"""Prompt templates for system, solve, repair, and tool calling modes."""

from __future__ import annotations

import json
from typing import Any

from physics_reasoning.core.models import (
    LLMParsedOutput,
    VerificationError,
)

SYSTEM_PROMPT_TEMPLATE = """You are an expert physics problem analyst. Your task is to analyze physics word problems (in Vietnamese or English) and extract structured physical concepts, quantities, and equations.

CRITICAL INSTRUCTIONS:
1. You do NOT need to perform manual arithmetic calculations. The deterministic symbolic solver (SymPy) will compute the mathematical solution.
2. Identify all GIVEN quantities from the problem with their numeric values and units:
   - Density / Khối lượng riêng: symbol "rho" (or "D"), unit "kg/m**3" (Formula: rho = m / V or rho = m / (l * w * h))
   - Mass / Khối lượng: symbol "m", unit "kg"
   - Dimensions: length "l" (m), width "w" (m), height "h" (m), volume "V" (m**3) (Formula: V = l * w * h)
   - Kinematics: acceleration "a" (m/s**2), velocity "v" (m/s), initial velocity "v_i" (m/s), displacement "d" / "h" (m), time "t" (s)
   - Electricity: voltage / Hiệu điện thế "U" (V), resistance / Điện trở "R" (ohm), current / Cường độ dòng điện "I" (A)
   - Pressure / Áp suất: symbol "p" (Pa), Force / Lực "F" (N), Area / Diện tích "S" (m**2)
   - Mechanical Work & Power: Power "P" (W), Work "A" (J) (Formula: P = (m * g * h) / t or P = (F * h) / t or A = m * g * h)
   - Thermodynamics / Cân bằng nhiệt: mass "m_1", "m_2", initial temperatures "t_1", "t_2", equilibrium temperature "t_cb" (Formula: m_1 * (t_cb - t_1) + m_2 * (t_cb - t_2) = 0. Always pair mass 1 with initial temperature 1, and mass 2 with initial temperature 2!)
3. Identify the EXACT TARGET quantity asked in the question (role: "target", symbol, and unit). Note: "Khối lượng riêng" is density ("rho"), NOT mass ("m")!
4. Select the standard physical equation(s) connecting the given quantities to the target.
5. Always return a valid JSON object matching the requested schema.

{available_equations_section}
"""

JSON_SCHEMA_EXAMPLE = """{
  "problem_understanding": "<Brief summary of what the problem gives and asks>",
  "quantities": [
    {"name": "<quantity_name>", "symbol": "<symbol>", "value": 10.0, "unit": "<unit_string>", "role": "given"},
    {"name": "<target_name>", "symbol": "<target_symbol>", "value": null, "unit": "<unit_string>", "role": "target"}
  ],
  "equations": [
    {"equation_id": "<equation_id>", "expression": "<variable = formula>", "justification": "<why this equation applies>"}
  ],
  "target_variable": "<target_symbol>",
  "solution_steps": [
    "Identify knowns and unknowns",
    "Apply relevant equations"
  ],
  "proposed_answer": null,
  "proposed_unit": "<unit_of_target>"
}"""


def build_system_prompt(available_equations: list[str] | None = None) -> str:
    """Build the system prompt for the LLM."""
    if available_equations:
        eq_list = "\n".join(f"- {eq}" for eq in available_equations[:50])
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
1. Identify the core physical principle/law:
   - Quán tính (inertia_law): vật duy trì vận tốc khi có lực hãm/tăng tốc đột ngột.
   - Áp suất chất rắn (solid_pressure): p = F/S, diện tích tiếp xúc S nhỏ thì áp suất p lớn (dao sắc, đinh nhọn).
   - Sự nhiễm điện do cọ xát (electrostatic_charging_friction): Khi quay cọ xát với không khí, cánh quạt bị nhiễm điện và hút các vật nhỏ nhẹ như hạt bụi.
   - Lực đẩy Ác-si-mét (archimedes_buoyant_force): F_A = d * V, điều kiện chìm/nổi do d_vat so với d_long.
   - Lực ma sát (friction_mechanisms): ma sát trượt, lăn, nghỉ cản trở chuyển động.
   - Sự bay hơi thu nhiệt (evaporation_cooling): chất lỏng bay hơi lấy nhiệt làm mát môi trường xung quanh.
   - Dẫn nhiệt (thermal_conduction) & Sự dãn nở nhiệt (thermal_expansion_uneven).
2. Form a complete, unbroken Causal Chain:
   - Step 1: Initial State & External Action (Trạng thái ban đầu và tác động/thay đổi xảy ra).
   - Step 2: Physical Mechanism (Cơ chế vật lý theo định luật: tại sao các bộ phận phản ứng khác nhau do tính chất vật lý).
   - Step 3: Observed Consequence (Hệ quả cuối cùng giải thích đúng hiện tượng được hỏi).
3. AVOID COMMON MISCONCEPTIONS:
   - NEVER refer to "inertia" as a force (quán tính không phải là lực đẩy/kéo).
   - Do NOT confuse force (áp lực F) with pressure (áp suất p = F/S).
   - Do NOT confuse heat (nhiệt lượng) with temperature (nhiệt độ).
   - Do NOT claim dust sticks solely because of friction (cánh quạt bám bụi là do cọ xát tạo ra sự NHIỄM ĐIỆN hút bụi nhẹ).
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
