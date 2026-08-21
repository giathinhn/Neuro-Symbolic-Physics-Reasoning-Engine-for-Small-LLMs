"""Output parser with robust extraction strategies for LLM-generated JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from physics_reasoning.core.enums import QuantityRole
from physics_reasoning.core.exceptions import LLMOutputParseError
from physics_reasoning.core.models import (
    LLMParsedOutput,
    ParsedEquation,
    ParsedQuantity,
    QualitativeParsedOutput,
)


def _extract_json_block(text: str) -> str | None:
    """Extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_outermost_braces(text: str) -> str | None:
    """Extract substring from the first '{' to the last '}'."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _normalize_parsed_dict(data: Any, raw_text: str = "") -> dict[str, Any]:
    """Normalize fields in raw parsed dict to be resilient against LLM quirks."""
    if not isinstance(data, dict):
        return {}

    # Handle proposed_answer as dict {"value": 2.0, "unit": "A"} or string "2.0"
    if "proposed_answer" in data:
        pa = data["proposed_answer"]
        if isinstance(pa, dict):
            if "unit" in pa and not data.get("proposed_unit"):
                data["proposed_unit"] = str(pa["unit"])
            val = pa.get("value")
            if val is not None:
                try:
                    data["proposed_answer"] = float(val)
                except Exception:
                    data["proposed_answer"] = None
            else:
                data["proposed_answer"] = None
        elif isinstance(pa, str):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", pa)
            if m:
                try:
                    data["proposed_answer"] = float(m.group(0))
                except Exception:
                    data["proposed_answer"] = None
            else:
                data["proposed_answer"] = None

    if "quantities" not in data or not isinstance(data["quantities"], list):
        data["quantities"] = []

    # If data had top-level keys like {"current": 2, "resistance": 12}, extract to quantities
    for k, v in list(data.items()):
        if k not in (
            "problem_understanding",
            "quantities",
            "equations",
            "target_variable",
            "solution_steps",
            "proposed_answer",
            "proposed_unit",
            "status",
        ):
            if isinstance(v, (int, float)):
                data["quantities"].append(
                    {
                        "name": str(k),
                        "symbol": str(k)[0],
                        "value": float(v),
                        "unit": "",
                        "role": "intermediate",
                    }
                )

    target_var = data.get("target_variable")
    for q in data["quantities"]:
        if isinstance(q, dict):
            if isinstance(q.get("value"), str):
                val_str = q["value"].replace(",", ".").strip()
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val_str)
                if m:
                    try:
                        q["value"] = float(m.group(0))
                    except Exception:
                        q["value"] = None
                else:
                    q["value"] = None

            role = str(q.get("role", "")).lower().strip()
            if role in ("given", "known"):
                q["role"] = "given"
            elif role in ("target", "to_find", "find", "solve_for"):
                q["role"] = "target"
            elif role in ("unknown", "intermediate", "calculated"):
                if target_var and q.get("symbol") == target_var:
                    q["role"] = "target"
                else:
                    q["role"] = "intermediate"
            elif not role:
                if target_var and q.get("symbol") == target_var:
                    q["role"] = "target"
                elif q.get("value") is not None:
                    q["role"] = "given"
                else:
                    q["role"] = "intermediate"

    if "equations" not in data or not isinstance(data["equations"], list):
        data["equations"] = []

    # If equations list is empty, search raw_text for equations (e.g. V = I * R or I = V / R or F = m * a)
    if not data["equations"] and raw_text:
        eq_patterns = re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Za-z0-9_+\-*/^. ()]+)", raw_text
        )
        for eq_str in eq_patterns:
            eq_clean = eq_str.strip()
            if "=" in eq_clean and len(eq_clean) > 3:
                data["equations"].append(
                    {
                        "equation_id": None,
                        "expression": eq_clean,
                        "justification": "Extracted from LLM response text",
                    }
                )

    return data


def parse_llm_output(raw_output: str) -> LLMParsedOutput:
    """Parse raw LLM string into validated LLMParsedOutput model."""
    if not raw_output or not raw_output.strip():
        raise LLMOutputParseError("LLM output is empty", raw_output=raw_output)

    clean_text = raw_output.strip()

    # Strategy 1: Direct JSON parse
    try:
        data = json.loads(clean_text)
        return LLMParsedOutput.model_validate(_normalize_parsed_dict(data, clean_text))
    except Exception:
        pass

    # Strategy 2: Extract code block
    code_block = _extract_json_block(clean_text)
    if code_block:
        try:
            data = json.loads(code_block)
            return LLMParsedOutput.model_validate(_normalize_parsed_dict(data, clean_text))
        except Exception:
            pass

    # Strategy 3: Outermost curly braces
    outer_braces = _extract_outermost_braces(clean_text)
    if outer_braces:
        try:
            data = json.loads(outer_braces)
            return LLMParsedOutput.model_validate(_normalize_parsed_dict(data, clean_text))
        except Exception:
            pass

    # Strategy 4: Resilient Fallback - extract formulas and values directly from text
    eq_matches = re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Za-z0-9_+\-*/^. ()]+)", clean_text
    )
    equations = [
        ParsedEquation(equation_id=None, expression=eq.strip(), justification="Text fallback")
        for eq in eq_matches
        if "=" in eq and len(eq.strip()) > 3
    ]

    if not equations:
        raise LLMOutputParseError(
            "Failed to parse LLM output as JSON or extract equations from text",
            raw_output=raw_output,
        )

    return LLMParsedOutput(
        problem_understanding="Parsed via text fallback strategy",
        quantities=[],
        equations=equations,
        target_variable="",
        solution_steps=[],
        proposed_answer=None,
        proposed_unit=None,
    )


def validate_parsed_output(output: LLMParsedOutput) -> list[str]:
    """Perform semantic sanity checks on parsed output."""
    warnings: list[str] = []

    target_quantities = [q for q in output.quantities if q.role == QuantityRole.TARGET]
    if not target_quantities and not output.target_variable:
        warnings.append("No target quantity or target_variable specified in output.")

    if not output.equations:
        warnings.append("No equations provided in output.")

    for eq in output.equations:
        if "=" not in eq.expression:
            warnings.append(f"Equation '{eq.expression}' is missing '='.")

    return warnings


def parse_qualitative_llm_output(raw_output: str) -> QualitativeParsedOutput:
    """Parse raw LLM output into validated QualitativeParsedOutput model."""
    if not raw_output or not raw_output.strip():
        raise LLMOutputParseError("LLM qualitative output is empty", raw_output=raw_output)

    clean_text = raw_output.strip()

    # Strategy 1: Direct JSON parse
    try:
        data = json.loads(clean_text)
        return QualitativeParsedOutput.model_validate(data)
    except Exception:
        pass

    # Strategy 2: Code fence
    code_block = _extract_json_block(clean_text)
    if code_block:
        try:
            data = json.loads(code_block)
            return QualitativeParsedOutput.model_validate(data)
        except Exception:
            pass

    # Strategy 3: Outermost braces
    outer_braces = _extract_outermost_braces(clean_text)
    if outer_braces:
        try:
            data = json.loads(outer_braces)
            return QualitativeParsedOutput.model_validate(data)
        except Exception:
            pass

    # Fallback: Create structured QualitativeParsedOutput from raw text
    return QualitativeParsedOutput(
        problem_understanding="Trực tiếp từ văn bản phản hồi",
        observed_phenomenon=clean_text[:100],
        core_principles=[],
        causal_chain=[],
        conclusion=clean_text,
        scientific_keywords=[],
    )
