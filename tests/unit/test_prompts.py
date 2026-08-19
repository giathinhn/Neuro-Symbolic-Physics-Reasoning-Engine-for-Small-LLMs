"""Tests for prompt templates."""

from __future__ import annotations

from physics_reasoning.core.enums import ErrorSeverity, ErrorType
from physics_reasoning.core.models import (
    LLMParsedOutput,
    ParsedEquation,
    ParsedQuantity,
    VerificationError,
)
from physics_reasoning.llm.prompts import (
    build_repair_prompt,
    build_solve_prompt,
    build_system_prompt,
    build_tool_call_prompt,
)


class TestPrompts:
    def test_build_system_prompt(self):
        prompt = build_system_prompt(["Newton's Second Law (F = m * a)"])
        assert "Newton's Second Law" in prompt
        assert "SymPy" in prompt

    def test_build_solve_prompt(self):
        prompt = build_solve_prompt("A 2 kg object experiences a force of 10 N.")
        assert "A 2 kg object" in prompt
        assert "JSON" in prompt

    def test_build_repair_prompt(self):
        parsed = LLMParsedOutput(
            quantities=[ParsedQuantity(name="force", symbol="F", value=10.0)],
            equations=[ParsedEquation(expression="F = m * t")],
        )
        errors = [
            VerificationError(
                error_type=ErrorType.DIMENSION_MISMATCH,
                severity=ErrorSeverity.ERROR,
                message="LHS has dimension M L T^-2 but RHS has M T",
                suggestion="Use F = m * a",
            )
        ]
        prompt = build_repair_prompt(parsed, errors, attempt_number=1)
        assert "DIMENSION_MISMATCH" in prompt
        assert "Use F = m * a" in prompt
        assert "F = m * t" in prompt

    def test_build_tool_call_prompt(self):
        prompt = build_tool_call_prompt("Calculate kinetic energy.")
        assert "search_equations" in prompt
        assert "calculate" in prompt
