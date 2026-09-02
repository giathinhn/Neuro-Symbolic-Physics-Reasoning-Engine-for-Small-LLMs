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
   - Density / Khối lượng riêng: symbol "rho" (or "D"), unit "kg/m**3" (Formula: rho = m / V)
   - Specific weight / Trọng lượng riêng: symbol "d", unit "N/m**3" (Formula: p = d * h, F_A = d * V, d = 10 * rho)
   - Mass / Khối lượng: symbol "m", unit "kg" (or "g")
   - Dimensions: length "l" (m), width "w" (m), height "h" (m), volume "V" (m**3)
   - Kinematics: acceleration "a" (m/s**2), velocity "v" (m/s), initial velocity "v_i" (m/s), displacement "s" / "h" (m), time "t" (s)
   - Electricity:
     * Ohm's law: I = U / R, I_1 = U_1 / R_1
     * Power & Resistance: P = U * I, P = I^2 * R, P = U^2 / R (or R = U^2 / P)
     * Series circuit: R_eq = R_1 + R_2 (or R_eq = R_1 + R_2 + R_3)
     * Parallel circuit: 1/R_eq = 1/R_1 + 1/R_2 (or 1/R_eq = 1/R_1 + 1/R_2 + 1/R_3)
     * Wire resistance: R = (rho * l) / S
     * Energy / Heat: A = P * t, Q = (U^2 / R) * t, Q = I^2 * R * t
   - Pressure & Fluids:
     * Solid pressure: p = F / S (or p = (m * g) / S)
     * Liquid pressure: p = d * h (or p = rho * g * h)
     * Archimedes buoyancy: F_A = d * V (or F_A = rho * g * V)
     * Hydraulic press: F_2 / F_1 = S_2 / S_1 (or F_2 = F_1 * (S_2 / S_1))
   - Mechanical Work & Power: Power P = A / t, Work A = F * s (or A = m * g * h), Efficiency H = A_ich / A_tp
   - Thermodynamics / Heat:
     * Heating: Q = m * c * (t_2 - t_1) (When water is boiled from t1, boiling temperature is implicitly t2 = 100°C!)
     * Fuel burning: Q = q * m
     * Heat balance: m_1 * c_1 * (t_cb - t_1) + m_2 * c_2 * (t_cb - t_2) = 0
3. Identify the EXACT TARGET quantity asked in the question (role: "target", symbol, and unit).
4. Write standard physical equations without manual conversion factors (do NOT write "/ 60" or "/ 1000" in equations; write standard formulas like A = P * t, Q = (U^2 / R) * t).
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
1. Identify the EXACT core physical principle/law:
   - Quán tính (inertia_law): vật có xu hướng duy trì vận tốc; khi phanh gấp người ngã chúi về PHÍA TRƯỚC, khi rẽ phải người nghiêng sang TRÁI.
   - Áp suất chất rắn (solid_pressure_area): p = F/S; diện tích S nhỏ thì áp suất p lớn (mũi kim, đinh nhọn, dao sắc); diện tích S lớn thì áp suất p nhỏ tránh lún (bánh xích, móng nhà).
   - Áp suất khí quyển (atmospheric_pressure_effects): khí quyển tác dụng áp suất lên mọi vật; khi hút bớt khí bên trong thì áp suất khí quyển bên ngoài ép bẹp hộp sữa hoặc đẩy nước lên trong ống hút; nắp ấm có lỗ để cân bằng áp suất.
   - Bình thông nhau (communicating_vessels_principle): các nhánh cùng một chất lỏng đứng yên có mặt thoáng ở cùng độ cao; vòi ấm đun nước phải cao ngang miệng ấm.
   - Sự nhiễm điện do cọ xát (electrostatic_charging_friction): khi cọ xát với không khí, cánh quạt bị nhiễm điện nên sinh lực tĩnh điện hút các hạt bụi nhỏ nhẹ.
   - Lực đẩy Ác-si-mét & Chìm nổi (archimedes_buoyancy_floating): F_A = d * V; vật nổi khi trọng lượng riêng trung bình d_v < d_l (tàu thủy rỗng nên d_tb < d_nuoc).
   - Lực ma sát (friction_mechanisms): ma sát trượt, lăn, nghỉ cản trở chuyển động; khía rãnh lốp xe tăng ma sát; tra dầu mỡ giảm ma sát.
   - Dãn nở nhiệt không đồng đều (thermal_expansion_unequal): rót nước sôi vào cốc dày dễ vỡ vì thủy tinh dẫn nhiệt kém, lớp trong nở ra trước khi lớp ngoài kịp nở gây ứng suất nứt vỡ.
   - Dẫn nhiệt (heat_conduction_rate): kim loại dẫn nhiệt nhanh hơn gỗ nên mùa đông sờ vào kim loại thấy lạnh hơn.
   - Sự bay hơi thu nhiệt (evaporation_heat_absorption): bay hơi luôn thu nhiệt từ bề mặt tiếp xúc làm giảm nhiệt độ và mát da (cồn, mồ hôi).
   - Sự ngưng tụ của hơi nước (condensation_dew_formation): hơi nước trong không khí gặp bề mặt lạnh bị ngưng tụ thành giọt nước (cốc nước đá, sương mù, mờ gương).
   - Đối lưu & Bức xạ nhiệt (convection_thermal_radiation): khí nóng nhẹ bay lên, khí lạnh nặng chìm xuống tạo dòng đối lưu (điều hòa lắp trên cao, lò sưởi dưới sàn).
   - Truyền thẳng ánh sáng (light_rectilinear_propagation): ánh sáng truyền thẳng trong môi trường trong suốt đồng tính (bóng tối, nhật thực, nguyệt thực).
   - Phản xạ ánh sáng & Gương (light_reflection_mirrors): gương cầu lồi có thị trường (vùng nhìn thấy) rộng hơn gương phẳng (gương chiếu hậu ô tô, gương khúc cua).
   - Khúc xạ ánh sáng (light_refraction_phenomena): ánh sáng đổi hướng qua mặt phân cách làm ảnh ảo của đáy hồ/bể bơi bị nâng cao lên (trông nông hơn thực tế), đũa trong nước trông như bị gãy.
   - Nguồn âm & Dao động (sound_source_vibration): vật phát ra âm khi đang dao động (mặt trống, dây đàn).
   - Sự truyền âm (sound_propagation_media): âm truyền tốt trong chất rắn, lỏng, khí (v_rắn > v_lỏng > v_khí); KHÔNG TRUYỀN trong chân không.
   - Phản xạ âm & Tiếng vang (sound_reflection_echo): nghe tiếng vang khi âm phản xạ đến sau âm trực tiếp ít nhất 1/15s; vật mềm xốp hấp thụ âm tốt dùng cách âm.

2. Form a complete, unbroken Causal Chain (at least 2-3 steps):
   - Step 1: Initial State & External Action (Trạng thái ban đầu và tác động/thay đổi xảy ra).
   - Step 2: Physical Mechanism (Cơ chế vật lý theo định luật: tại sao các bộ phận phản ứng khác nhau do tính chất vật lý).
   - Step 3: Observed Consequence (Hệ quả cuối cùng giải thích đúng hiện tượng được hỏi).

3. AVOID COMMON MISCONCEPTIONS:
   - NEVER refer to "inertia" as a force (quán tính không phải là lực đẩy/kéo).
   - Do NOT confuse force (áp lực F) with pressure (áp suất p = F/S).
   - Do NOT confuse heat (nhiệt lượng) with temperature (nhiệt độ).
   - Do NOT claim dust sticks solely because of friction (cánh quạt bám bụi là do cọ xát tạo ra sự NHIỄM ĐIỆN hút bụi nhẹ).
   - Do NOT claim objects sink solely because they are heavy (chìm/nổi do trọng lượng riêng so với chất lỏng).
   - Do NOT claim water outside an iced glass leaked through the glass (là do hơi nước trong không khí NGƯNG TỤ khi gặp lạnh).
   - Do NOT claim the stick in water physically snapped (là do KHÚC XẠ ÁNH SÁNG tạo ảnh ảo bị lệch).
   - Do NOT claim a vacuum has a natural suction force (là do ÁP SUẤT KHÍ QUYỂN bên ngoài đẩy vào).

4. Always respond with a structured JSON object matching the requested schema.

{available_principles_section}
"""

QUALITATIVE_JSON_SCHEMA_EXAMPLE = """{
  "problem_understanding": "Tóm tắt hiện tượng cần giải thích",
  "observed_phenomenon": "Hiện tượng được quan sát trong thực tế",
  "core_principles": ["<id_nguyen_ly_phu_hop_tu_danh_sach>"],
  "causal_chain": [
    {
      "step_number": 1,
      "state_or_action": "Trạng thái ban đầu và tác động/thay đổi xảy ra.",
      "physical_mechanism": "Cơ chế tác động vật lý ban đầu theo định luật.",
      "governing_principle": "<id_nguyen_ly>"
    },
    {
      "step_number": 2,
      "state_or_action": "Phản ứng của hệ vật hoặc các bộ phận liên quan.",
      "physical_mechanism": "Cơ chế vật lý chi tiết giải thích sự khác biệt.",
      "governing_principle": "<id_nguyen_ly>"
    },
    {
      "step_number": 3,
      "state_or_action": "Kết quả tổng thể của hiện tượng.",
      "physical_mechanism": "Hệ quả quan sát được trả lời trực tiếp câu hỏi bài toán.",
      "governing_principle": "<id_nguyen_ly>"
    }
  ],
  "conclusion": "Câu kết luận khoa học cô đọng giải thích đầy đủ và chính xác hiện tượng.",
  "scientific_keywords": ["từ_khóa_1", "từ_khóa_2"]
}"""


def build_qualitative_system_prompt(available_principles: list[str] | None = None) -> str:
    """Build the system prompt for qualitative physics phenomenon explanations."""
    if available_principles:
        p_list = "\n".join(f"- {p}" for p in available_principles[:15])
        section = f"Standard qualitative physics principles available in knowledge base:\n{p_list}"
    else:
        section = ""
    return QUALITATIVE_SYSTEM_PROMPT_TEMPLATE.format(available_principles_section=section)


def build_qualitative_solve_prompt(
    problem_text: str, relevant_principles: list[str] | None = None
) -> str:
    """Build user prompt for explaining a qualitative physics phenomenon."""
    principle_hints = ""
    if relevant_principles:
        hints_str = "\n".join(f"  * {p}" for p in relevant_principles)
        principle_hints = f"\nCANDIDATE PRINCIPLES FOR THIS PROBLEM:\n{hints_str}\n(Select the most appropriate principle ID from above for 'core_principles')\n"

    return f"""Please provide a rigorous, scientifically grounded explanation for the following physics phenomenon:

PHENOMENON QUESTION:
"{problem_text}"
{principle_hints}
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
