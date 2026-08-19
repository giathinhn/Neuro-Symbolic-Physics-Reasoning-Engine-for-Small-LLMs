"""FastAPI application for the Physics Reasoning Engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from physics_reasoning.api.schemas import HealthResponse, SolveRequest, SolveResponse
from physics_reasoning.core.config import PipelineConfig, load_config
from physics_reasoning.llm.provider import LiteLLMProvider
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

# Global pipeline instance
_ORCHESTRATOR: PipelineOrchestrator | None = None
_KB: KnowledgeBase | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize knowledge base and engine on startup."""
    global _ORCHESTRATOR, _KB
    config = load_config()
    _KB = KnowledgeBase(config.knowledge_base_path)
    _KB.load()
    _ORCHESTRATOR = PipelineOrchestrator(config=config, knowledge_base=_KB)
    yield


app = FastAPI(
    title="Neuro-Symbolic Physics Reasoning Engine",
    description="Tool-augmented reasoning engine for small LLMs to reliably solve physics problems.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    global _KB
    if _KB is None:
        config = load_config()
        _KB = KnowledgeBase(config.knowledge_base_path)
        _KB.load()
    eq_count = len(_KB.equations) if _KB else 0
    q_count = len(_KB.quantities) if _KB else 0
    return HealthResponse(
        status="ok",
        version="0.1.0",
        equations_count=eq_count,
        quantities_count=q_count,
    )


@app.get("/equations")
async def list_equations() -> list[dict[str, Any]]:
    """List all equations available in the physics knowledge base."""
    global _KB
    if _KB is None:
        config = load_config()
        _KB = KnowledgeBase(config.knowledge_base_path)
        _KB.load()
    return [eq.model_dump() for eq in _KB.equations.values()]


@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest) -> SolveResponse:
    """Solve a physics problem with the neuro-symbolic engine."""
    global _ORCHESTRATOR, _KB
    if not _ORCHESTRATOR:
        config = load_config()
        _KB = KnowledgeBase(config.knowledge_base_path)
        _KB.load()
        _ORCHESTRATOR = PipelineOrchestrator(config=config, knowledge_base=_KB)

    # Optional overrides
    if request.model or request.max_retries is not None:
        cfg = _ORCHESTRATOR.config.model_copy()
        if request.model:
            cfg.model_name = request.model
        if request.max_retries is not None:
            cfg.max_retries = request.max_retries
        custom_llm = LiteLLMProvider(model_name=cfg.model_name)
        orchestrator = PipelineOrchestrator(
            config=cfg,
            llm_provider=custom_llm,
            knowledge_base=_KB,
        )
    else:
        orchestrator = _ORCHESTRATOR

    try:
        solution = orchestrator.solve(request.problem)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solving failed: {e}") from e

    errors_list = None
    if solution.verification_result and solution.verification_result.errors:
        errors_list = [
            {
                "error_type": e.error_type.value,
                "severity": e.severity.value,
                "message": e.message,
                "suggestion": e.suggestion or "",
            }
            for e in solution.verification_result.errors
        ]

    verification_summary = {}
    if solution.verification_result:
        verification_summary = {
            "is_valid": solution.verification_result.is_valid,
            "confidence": solution.verification_result.confidence,
            "checks_passed": solution.verification_result.checks_passed,
            "checks_performed": solution.verification_result.checks_performed,
        }

    return SolveResponse(
        answer=solution.answer_value,
        unit=solution.answer_unit,
        is_verified=solution.is_verified,
        equations_used=solution.equations_used,
        quantities=[q.model_dump() for q in solution.quantities_extracted],
        verification=verification_summary,
        reasoning_steps=[s.action for s in solution.solve_steps],
        metadata={
            "attempts": solution.num_attempts,
            "tokens": solution.total_tokens,
            "llm_calls": solution.total_llm_calls,
            "latency_ms": solution.latency_ms,
        },
        errors=errors_list,
    )
