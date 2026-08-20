"""Solve context for tracking problem state and attempts across iterations."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from physics_reasoning.core.models import (
    SolveStep,
    ToolCallRecord,
    VerificationError,
)


class SolveContext:
    """Tracks state and history during solving of a single physics problem."""

    def __init__(
        self,
        problem_text: str,
        problem_id: str | None = None,
        max_tool_calls_per_attempt: int = 10,
    ):
        self.problem_id: str = problem_id or str(uuid4())
        self.problem_text: str = problem_text
        self.attempt: int = 0
        self.max_tool_calls_per_attempt: int = max_tool_calls_per_attempt

        self.messages: list[dict[str, str]] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.solve_steps: list[SolveStep] = []
        self.errors_by_attempt: list[list[VerificationError]] = []

        self.total_tokens: int = 0
        self.total_llm_calls: int = 0
        self.start_time: datetime = datetime.now()

    def add_step(
        self,
        action: str,
        input_data: dict | None = None,
        output_data: dict | None = None,
        tool_name: str | None = None,
        duration_ms: float = 0.0,
        output: dict | None = None,
    ) -> None:
        """Record a pipeline execution step."""
        actual_output = output_data if output_data is not None else (output or {})
        actual_input = input_data or {}
        step = SolveStep(
            step_number=len(self.solve_steps) + 1,
            action=action,
            input_data=actual_input,
            output_data=actual_output,
            tool_name=tool_name,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
        )
        self.solve_steps.append(step)
