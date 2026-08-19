"""Main Pipeline Orchestrator coordinating LLM, symbolic solving, and verification."""

from __future__ import annotations

import time
from typing import Any

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.core.enums import QuantityRole
from physics_reasoning.core.exceptions import LLMError, LLMOutputParseError
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    PhysicsQuantity,
    Solution,
    SolveResult,
    VerificationResult,
)
from physics_reasoning.llm.output_parser import parse_llm_output
from physics_reasoning.llm.prompts import (
    build_repair_prompt,
    build_solve_prompt,
    build_system_prompt,
)
from physics_reasoning.llm.provider import LLMProvider, LiteLLMProvider
from physics_reasoning.physics.constants import PHYSICAL_CONSTANTS, PHYSICAL_CONSTANT_UNITS
from physics_reasoning.physics.equation_retriever import EquationRetriever
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.physics.quantity_extractor import QuantityExtractor
from physics_reasoning.pipeline.context import SolveContext
from physics_reasoning.solver.symbolic_solver import SymbolicSolver
from physics_reasoning.units.dimension_checker import DimensionChecker
from physics_reasoning.units.unit_engine import UnitEngine
from physics_reasoning.verifier.verification_pipeline import VerificationPipeline


class PipelineOrchestrator:
    """Orchestrate the neuro-symbolic physics reasoning pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        llm_provider: LLMProvider | None = None,
        knowledge_base: KnowledgeBase | None = None,
        solver: SymbolicSolver | None = None,
        unit_engine: UnitEngine | None = None,
        dimension_checker: DimensionChecker | None = None,
        verification_pipeline: VerificationPipeline | None = None,
    ):
        self.config = config or PipelineConfig()
        self.llm = llm_provider or LiteLLMProvider(model_name=self.config.model_name)

        self.kb = knowledge_base or KnowledgeBase(self.config.knowledge_base_path)
        # Attempt to load KB if not already loaded
        if not self.kb.equations:
            try:
                self.kb.load()
            except Exception:
                pass

        self.solver = solver or SymbolicSolver(
            numerical_tolerance=self.config.numerical_tolerance,
            timeout_seconds=self.config.solver_timeout_seconds,
        )
        self.unit_engine = unit_engine or UnitEngine()
        self.dimension_checker = dimension_checker or DimensionChecker(self.unit_engine)
        self.retriever = EquationRetriever(self.kb)
        self.quantity_extractor = QuantityExtractor(self.kb, self.unit_engine)

        self.verifier = verification_pipeline or VerificationPipeline(
            config=self.config,
            solver=self.solver,
            unit_engine=self.unit_engine,
            dimension_checker=self.dimension_checker,
            knowledge_base=self.kb,
        )

    def solve(self, problem_text: str, problem_id: str | None = None) -> Solution:
        """Solve a physics word problem end-to-end with the verify-repair loop."""
        start_time = time.perf_counter()
        ctx = SolveContext(
            problem_text=problem_text,
            problem_id=problem_id,
            max_tool_calls_per_attempt=self.config.max_tool_calls_per_attempt,
        )

        # 1. Prepare initial conversation messages
        available_eq_names = [f"{eq.name} ({eq.expression})" for eq in self.kb.equations.values()]
        system_prompt = build_system_prompt(available_eq_names)
        user_prompt = build_solve_prompt(problem_text)

        ctx.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_parsed_output: LLMParsedOutput | None = None
        last_solve_result: SolveResult | None = None
        last_verification_result: VerificationResult | None = None
        last_equations_used: list[Equation] = []
        last_quantities_extracted: list[PhysicsQuantity] = []
        answer_value: float | None = None
        answer_unit: str | None = None

        while ctx.attempt < self.config.max_retries:
            ctx.attempt += 1
            attempt_start = time.perf_counter()

            # Call LLM
            try:
                llm_resp = self.llm.complete(
                    messages=ctx.messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                ctx.total_llm_calls += 1
                ctx.total_tokens += llm_resp.usage.get("total_tokens", 0)
                raw_content = llm_resp.content
            except Exception as e:
                ctx.add_step(
                    action="llm_call_failed",
                    input_data={"messages_count": len(ctx.messages)},
                    output_data={"error": str(e)},
                    duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
                )
                break

            # Append assistant response to history
            ctx.messages.append({"role": "assistant", "content": raw_content})

            # Parse LLM output
            try:
                parsed_output = parse_llm_output(raw_content)
                last_parsed_output = parsed_output
            except LLMOutputParseError as e:
                ctx.add_step(
                    action="parse_failed",
                    input_data={"raw_output": raw_content},
                    output_data={"error": str(e)},
                    duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
                )
                # Retry with formatting reminder
                ctx.messages.append(
                    {
                        "role": "user",
                        "content": "Your response was not valid JSON. Please respond ONLY with a valid JSON object matching the schema.",
                    }
                )
                continue

            # Standardize quantities
            quantities = self.quantity_extractor.standardize_all(parsed_output.quantities)
            last_quantities_extracted = quantities

            # Match equations against knowledge base
            equations_used: list[Equation] = []
            eq_expressions: list[str] = []
            for eq_item in parsed_output.equations:
                eq_expressions.append(eq_item.expression)
                matched_eq = None
                if eq_item.equation_id:
                    matched_eq = self.kb.get_equation(eq_item.equation_id)
                if not matched_eq:
                    matched_eq = self.retriever.match_expression(eq_item.expression)
                if matched_eq:
                    equations_used.append(matched_eq)
            last_equations_used = equations_used

            # Extract target variable
            target_var = parsed_output.target_variable
            if not target_var:
                target_q = next((q for q in quantities if q.is_target), None)
                if target_q:
                    target_var = target_q.symbol
                else:
                    target_var = "x"

            # Prepare known values converted to standard SI units
            known_values_si: dict[str, float] = {}
            var_units: dict[str, str] = dict(PHYSICAL_CONSTANT_UNITS)

            # Add standard physical constants (e.g. g = 9.8)
            for const_sym, const_val in PHYSICAL_CONSTANTS.items():
                known_values_si[const_sym] = const_val

            for q in quantities:
                if q.symbol:
                    if q.unit:
                        var_units[q.symbol] = q.unit
                    if q.value is not None and not q.is_target:
                        if q.unit:
                            try:
                                si_val, _ = self.unit_engine.to_si(q.value, q.unit)
                                known_values_si[q.symbol] = si_val
                            except Exception:
                                known_values_si[q.symbol] = float(q.value)
                        else:
                            known_values_si[q.symbol] = float(q.value)

            # Symbolic Solving
            solve_res = self.solver.solve_system(
                equations=eq_expressions,
                known_values=known_values_si,
                target_variable=target_var,
            )
            last_solve_result = solve_res

            # Determine answer unit
            target_q = next((q for q in quantities if q.is_target or q.symbol == target_var), None)
            target_unit = target_q.unit if target_q and target_q.unit else parsed_output.proposed_unit
            if not target_unit and target_q and target_q.si_unit:
                target_unit = target_q.si_unit

            # If solver succeeded with numeric answer
            if solve_res.is_numeric and solve_res.solutions:
                # Select positive solution if available (default for magnitudes)
                num_solutions = [s for s in solve_res.solutions if isinstance(s, (int, float))]
                positive_solutions = [s for s in num_solutions if s >= 0]
                selected_val = positive_solutions[0] if positive_solutions else num_solutions[0]

                # Convert to requested target unit if different from SI
                if target_unit:
                    try:
                        conv = self.unit_engine.convert(selected_val, target_q.si_unit or target_unit, target_unit)
                        answer_value = conv.to_value
                    except Exception:
                        answer_value = selected_val
                else:
                    answer_value = selected_val
                answer_unit = target_unit

            # Verification Check
            v_res = self.verifier.verify(
                parsed_output=parsed_output,
                solve_result=solve_res,
                equations_used=equations_used,
                var_units=var_units,
                all_values=known_values_si,
            )
            last_verification_result = v_res

            ctx.add_step(
                action="attempt_complete",
                input_data={"attempt": ctx.attempt, "equations": eq_expressions},
                output_data={
                    "is_valid": v_res.is_valid,
                    "errors_count": len(v_res.errors),
                    "answer": answer_value,
                },
                duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
            )

            # If verification passed -> exit loop
            if v_res.is_valid:
                break

            # If failed -> generate repair prompt and retry
            repair_prompt = build_repair_prompt(
                previous_output=parsed_output,
                verification_errors=v_res.errors,
                attempt_number=ctx.attempt,
            )
            ctx.messages.append({"role": "user", "content": repair_prompt})

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        is_verified = (
            last_verification_result is not None
            and last_verification_result.is_valid
            and answer_value is not None
        )

        error_message = None
        if not is_verified and last_verification_result and last_verification_result.errors:
            error_message = "; ".join(e.message for e in last_verification_result.errors[:3])

        reasoning_steps = (
            last_parsed_output.solution_steps if last_parsed_output else []
        )

        return Solution(
            problem_id=ctx.problem_id,
            answer_value=answer_value,
            answer_unit=answer_unit,
            equations_used=[eq.id for eq in last_equations_used] if last_equations_used else [eq.expression for eq in (last_parsed_output.equations if last_parsed_output else [])],
            quantities_extracted=last_quantities_extracted,
            solve_steps=ctx.solve_steps,
            verification_result=last_verification_result,
            num_attempts=ctx.attempt,
            total_llm_calls=ctx.total_llm_calls,
            total_tool_calls=len(ctx.tool_calls),
            total_tokens=ctx.total_tokens,
            latency_ms=total_latency_ms,
            is_verified=is_verified,
            error_message=error_message,
        )
