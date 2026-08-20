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
)


def _extract_json_block(text: str) -> str | None:
    """Extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)."""
    # Look for ```json ... ```
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


def parse_llm_output(raw_output: str) -> LLMParsedOutput:
    """Parse raw LLM string into validated LLMParsedOutput model.

    Extraction strategies (ordered by priority):
    1. Direct JSON parse on full text.
    2. Extract from markdown code fence (```json ... ```).
    3. Extract from outermost curly braces '{ ... }'.

    Args:
        raw_output: Raw string output from LLM.

    Returns:
        Validated LLMParsedOutput.

    Raises:
        LLMOutputParseError: If parsing fails under all strategies.
    """
    if not raw_output or not raw_output.strip():
        raise LLMOutputParseError("LLM output is empty", raw_output=raw_output)

    clean_text = raw_output.strip()

    # Strategy 1: Direct parse
    try:
        data = json.loads(clean_text)
        return LLMParsedOutput.model_validate(data)
    except Exception:
        pass

    # Strategy 2: Code block
    code_block = _extract_json_block(clean_text)
    if code_block:
        try:
            data = json.loads(code_block)
            return LLMParsedOutput.model_validate(data)
        except Exception:
            pass

    # Strategy 3: Outermost braces
    outer_braces = _extract_outermost_braces(clean_text)
    if outer_braces:
        try:
            data = json.loads(outer_braces)
            return LLMParsedOutput.model_validate(data)
        except Exception:
            pass

    raise LLMOutputParseError(
        f"Failed to parse LLM output into structured JSON schema: {raw_output[:200]}...",
        raw_output=raw_output,
    )


def validate_parsed_output(output: LLMParsedOutput) -> list[str]:
    """Perform semantic sanity checks on parsed output.

    Returns:
        List of warning messages (empty if all valid).
    """
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
    from physics_reasoning.core.models import QualitativeParsedOutput

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

