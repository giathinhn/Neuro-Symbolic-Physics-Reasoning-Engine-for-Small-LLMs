"""Repair prompt generator for feedback loop."""

from __future__ import annotations

from physics_reasoning.core.models import (
    LLMParsedOutput,
    VerificationResult,
)
from physics_reasoning.llm.prompts import build_repair_prompt
from physics_reasoning.pipeline.context import SolveContext


def generate_repair_prompt(
    context: SolveContext,
    verification_result: VerificationResult,
    parsed_output: LLMParsedOutput | str,
) -> str:
    """Generate structured repair prompt from verification errors."""
    return build_repair_prompt(
        previous_output=parsed_output,
        verification_errors=verification_result.errors,
        attempt_number=context.attempt,
    )
