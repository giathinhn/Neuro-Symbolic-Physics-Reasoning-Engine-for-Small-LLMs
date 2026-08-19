"""FastAPI request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    """Request payload for /solve endpoint."""

    problem: str = Field(..., description="Physics word problem text", json_schema_extra={"example": "A 2 kg object experiences a force of 10 N. Find its acceleration."})
    model: str | None = Field(None, description="Optional LLM model override (e.g. 'ollama/phi3:mini', 'gpt-4o-mini')")
    max_retries: int | None = Field(None, description="Optional max retries override (default: 3)")
    enable_verification: bool = Field(True, description="Whether to execute verification checks and repair loop")


class SolveResponse(BaseModel):
    """Response payload for /solve endpoint."""

    answer: float | None = Field(None, description="Calculated numerical answer")
    unit: str | None = Field(None, description="Physical unit of the answer")
    is_verified: bool = Field(False, description="Whether the answer passed all verification checks")
    equations_used: list[str] = Field(default_factory=list, description="IDs or expressions of equations used")
    quantities: list[dict[str, Any]] = Field(default_factory=list, description="Extracted physical quantities")
    verification: dict[str, Any] = Field(default_factory=dict, description="Summary of verification checks")
    reasoning_steps: list[str] = Field(default_factory=list, description="Step-by-step physical reasoning")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata (attempts, tokens, latency)")
    errors: list[dict[str, str]] | None = Field(None, description="Verification errors if verification failed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    equations_count: int = 0
    quantities_count: int = 0
