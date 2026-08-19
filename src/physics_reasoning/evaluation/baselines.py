"""Baseline implementations: Raw LLM and LLM + Calculator."""

from __future__ import annotations

import re
import time

from physics_reasoning.core.models import Solution
from physics_reasoning.llm.provider import LLMProvider
from physics_reasoning.solver.expression_parser import parse_expression
from physics_reasoning.solver.numerical import evaluate_numeric


class RawLLMBaseline:
    """Baseline 1: Raw LLM directly outputs natural language reasoning and answer."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def solve(self, problem_text: str, problem_id: str | None = None) -> Solution:
        start_time = time.perf_counter()

        prompt = (
            f"Solve the following physics word problem. Provide step-by-step reasoning "
            f"and state your final numerical answer and unit at the very end in the format: "
            f"'Final Answer: <number> <unit>'.\n\nProblem:\n{problem_text}"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            resp = self.llm.complete(messages=messages, temperature=0.1)
            content = resp.content
            tokens = resp.usage.get("total_tokens", 0)
        except Exception as e:
            return Solution(
                problem_id=problem_id or "",
                is_verified=False,
                error_message=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # Extract number from 'Final Answer: 5.0 m/s^2' or last number in text
        answer_val: float | None = None
        answer_unit: str | None = None

        match = re.search(
            r"Final\s*Answer\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z/\^0-9\*\-]+)?",
            content,
            flags=re.IGNORECASE,
        )
        if match:
            try:
                answer_val = float(match.group(1))
                answer_unit = match.group(2)
            except Exception:
                pass
        else:
            # Fallback: extract last number
            numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", content)
            if numbers:
                try:
                    answer_val = float(numbers[-1])
                except Exception:
                    pass

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return Solution(
            problem_id=problem_id or "",
            answer_value=answer_val,
            answer_unit=answer_unit,
            num_attempts=1,
            total_llm_calls=1,
            total_tokens=tokens,
            latency_ms=latency_ms,
            is_verified=False,
        )


class CalculatorBaseline:
    """Baseline 2: LLM with a calculator evaluation tool."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def solve(self, problem_text: str, problem_id: str | None = None) -> Solution:
        start_time = time.perf_counter()

        prompt = (
            f"Solve the following physics word problem. You can perform calculations using Python expressions.\n"
            f"Write your mathematical derivation and end with 'CALCULATE: <expression>' on a new line "
            f"to compute the final answer, followed by the unit.\n\nProblem:\n{problem_text}"
        )

        messages = [{"role": "user", "content": prompt}]
        try:
            resp = self.llm.complete(messages=messages, temperature=0.1)
            content = resp.content
            tokens = resp.usage.get("total_tokens", 0)
        except Exception as e:
            return Solution(
                problem_id=problem_id or "",
                is_verified=False,
                error_message=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        answer_val: float | None = None
        # Check for CALCULATE: 10 / 2
        calc_match = re.search(r"CALCULATE\s*:\s*([^\n\r]+)", content, flags=re.IGNORECASE)
        if calc_match:
            expr_str = calc_match.group(1).strip()
            try:
                parsed = parse_expression(expr_str)
                answer_val = evaluate_numeric(parsed)
            except Exception:
                pass

        if answer_val is None:
            # Fallback to direct regex
            numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", content)
            if numbers:
                try:
                    answer_val = float(numbers[-1])
                except Exception:
                    pass

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return Solution(
            problem_id=problem_id or "",
            answer_value=answer_val,
            num_attempts=1,
            total_llm_calls=1,
            total_tokens=tokens,
            latency_ms=latency_ms,
            is_verified=False,
        )
