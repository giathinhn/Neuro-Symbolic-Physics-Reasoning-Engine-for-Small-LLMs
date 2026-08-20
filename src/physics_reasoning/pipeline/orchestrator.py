"""Main Pipeline Orchestrator coordinating LLM, symbolic solving, and verification."""

from __future__ import annotations

import time
from typing import Any

from physics_reasoning.core.config import PipelineConfig, load_config
from physics_reasoning.core.enums import ProblemType, QuantityRole
from physics_reasoning.core.exceptions import LLMError, LLMOutputParseError
from physics_reasoning.core.models import (
    Equation,
    LLMParsedOutput,
    PhysicsQuantity,
    QualitativeParsedOutput,
    Solution,
    SolveResult,
    VerificationResult,
)
from physics_reasoning.llm.output_parser import (
    parse_llm_output,
    parse_qualitative_llm_output,
)
from physics_reasoning.llm.prompts import (
    build_qualitative_repair_prompt,
    build_qualitative_solve_prompt,
    build_qualitative_system_prompt,
    build_repair_prompt,
    build_solve_prompt,
    build_system_prompt,
)
from physics_reasoning.llm.provider import LLMProvider, LiteLLMProvider
from physics_reasoning.physics.constants import PHYSICAL_CONSTANTS, PHYSICAL_CONSTANT_UNITS
from physics_reasoning.physics.equation_retriever import EquationRetriever
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.physics.qualitative_kb import QualitativeKnowledgeBase
from physics_reasoning.physics.quantity_extractor import QuantityExtractor
from physics_reasoning.pipeline.classifier import ProblemClassifier
from physics_reasoning.pipeline.context import SolveContext
from physics_reasoning.solver.symbolic_solver import SymbolicSolver
from physics_reasoning.units.dimension_checker import DimensionChecker
from physics_reasoning.units.unit_engine import UnitEngine
from physics_reasoning.verifier.qualitative_verifier import QualitativeVerificationPipeline
from physics_reasoning.verifier.verification_pipeline import VerificationPipeline


class PipelineOrchestrator:
    """Orchestrate the neuro-symbolic physics reasoning pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        llm_provider: LLMProvider | None = None,
        knowledge_base: KnowledgeBase | None = None,
        qualitative_knowledge_base: QualitativeKnowledgeBase | None = None,
        solver: SymbolicSolver | None = None,
        unit_engine: UnitEngine | None = None,
        dimension_checker: DimensionChecker | None = None,
        verification_pipeline: VerificationPipeline | None = None,
        qualitative_verification_pipeline: QualitativeVerificationPipeline | None = None,
    ):
        self.config = config or load_config()
        self.llm = llm_provider or LiteLLMProvider(
            model_name=self.config.model_name,
            timeout=self.config.timeout_seconds,
        )

        self.kb = knowledge_base or KnowledgeBase(self.config.knowledge_base_path)
        if not self.kb.equations:
            try:
                self.kb.load()
            except Exception:
                pass

        self.qualitative_kb = qualitative_knowledge_base or QualitativeKnowledgeBase()
        if not self.qualitative_kb.principles:
            try:
                self.qualitative_kb.load()
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
        self.qualitative_verifier = (
            qualitative_verification_pipeline
            or QualitativeVerificationPipeline(self.qualitative_kb)
        )

    def solve(self, problem_text: str, problem_id: str | None = None) -> Solution:
        """Solve a physics word problem end-to-end (quantitative or qualitative)."""
        # Determine problem classification
        problem_type = ProblemClassifier.classify(problem_text)
        if problem_type == ProblemType.QUALITATIVE:
            return self._solve_qualitative(problem_text, problem_id)

        return self._solve_quantitative(problem_text, problem_id)

    def _solve_qualitative(
        self, problem_text: str, problem_id: str | None = None
    ) -> Solution:
        """Execute the Qualitative Neuro-Symbolic Reasoning pipeline."""
        start_time = time.perf_counter()
        ctx = SolveContext(
            problem_text=problem_text,
            problem_id=problem_id,
            max_tool_calls_per_attempt=self.config.max_tool_calls_per_attempt,
        )

        # Retrieve relevant principles from KB
        relevant_principles = self.qualitative_kb.find_relevant_principles(problem_text, top_k=5)
        p_names = [f"{p.name} ({p.id}): {p.description[:80]}..." for p in self.qualitative_kb.principles.values()]

        system_prompt = build_qualitative_system_prompt(p_names)
        user_prompt = build_qualitative_solve_prompt(problem_text)

        ctx.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_qualitative_output: QualitativeParsedOutput | None = None
        last_verification_result: VerificationResult | None = None

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
                    output={"error": str(e)},
                    duration_ms=(time.perf_counter() - attempt_start) * 1000,
                )
                return Solution(
                    problem_id=ctx.problem_id,
                    is_qualitative=True,
                    is_verified=False,
                    num_attempts=ctx.attempt,
                    total_llm_calls=ctx.total_llm_calls,
                    total_tokens=ctx.total_tokens,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    error_message=f"LLM API error: {e}",
                    solve_steps=ctx.solve_steps,
                )

            # Parse qualitative output
            try:
                qual_output = parse_qualitative_llm_output(raw_content)
                last_qualitative_output = qual_output
            except Exception as e:
                # Retry on unparseable format
                if ctx.attempt < self.config.max_retries:
                    ctx.messages.append({"role": "assistant", "content": raw_content})
                    ctx.messages.append(
                        {
                            "role": "user",
                            "content": f"Output failed to parse as JSON: {e}. Please return valid JSON.",
                        }
                    )
                    continue
                break

            # Verify qualitative explanation
            ver_result = self.qualitative_verifier.verify(problem_text, qual_output)
            last_verification_result = ver_result

            ctx.add_step(
                action=f"attempt_{ctx.attempt}_qualitative_verification",
                input_data={"principles": qual_output.core_principles},
                output_data={
                    "is_valid": ver_result.is_valid,
                    "errors_count": len(ver_result.errors),
                    "confidence": ver_result.confidence,
                },
                duration_ms=(time.perf_counter() - attempt_start) * 1000,
            )

            # If verified successfully, return immediately
            if ver_result.is_valid:
                return Solution(
                    problem_id=ctx.problem_id,
                    is_qualitative=True,
                    qualitative_output=qual_output,
                    principles_applied=qual_output.core_principles,
                    final_explanation=qual_output.conclusion,
                    solve_steps=ctx.solve_steps,
                    verification_result=ver_result,
                    num_attempts=ctx.attempt,
                    total_llm_calls=ctx.total_llm_calls,
                    total_tokens=ctx.total_tokens,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    is_verified=True,
                )

            # Verify-Repair Loop: formulate targeted qualitative repair prompt
            if ctx.attempt < self.config.max_retries:
                repair_prompt = build_qualitative_repair_prompt(
                    previous_output=qual_output.model_dump_json(indent=2),
                    verification_errors=ver_result.errors,
                    attempt_number=ctx.attempt,
                )
                ctx.messages.append({"role": "assistant", "content": raw_content})
                ctx.messages.append({"role": "user", "content": repair_prompt})

        # Return best unverified qualitative solution
        return Solution(
            problem_id=ctx.problem_id,
            is_qualitative=True,
            qualitative_output=last_qualitative_output,
            principles_applied=(
                last_qualitative_output.core_principles if last_qualitative_output else []
            ),
            final_explanation=(
                last_qualitative_output.conclusion if last_qualitative_output else None
            ),
            solve_steps=ctx.solve_steps,
            verification_result=last_verification_result,
            num_attempts=ctx.attempt,
            total_llm_calls=ctx.total_llm_calls,
            total_tokens=ctx.total_tokens,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            is_verified=False,
            error_message="Max repair attempts exceeded for qualitative verification",
        )

    def _solve_quantitative(
        self, problem_text: str, problem_id: str | None = None
    ) -> Solution:
        """Solve a quantitative physics word problem end-to-end."""
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
            target_q = next((q for q in quantities if q.symbol == target_var), None)
            if not target_q:
                target_q = next((q for q in quantities if q.is_target), None)
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

            # Fill in missing variable units from KnowledgeBase
            from physics_reasoning.solver.expression_parser import extract_symbol_names
            for eq_expr in eq_expressions:
                for sym in extract_symbol_names(eq_expr):
                    if sym not in var_units:
                        kb_q = self.kb.get_quantity_by_symbol(sym)
                        if kb_q and kb_q.si_unit:
                            var_units[sym] = kb_q.si_unit

            # Verification Check
            all_values_for_verification = dict(known_values_si)
            if answer_value is not None and target_var:
                all_values_for_verification[target_var] = selected_val

            # If intermediate variables were solved, also evaluate them
            if solve_res.is_numeric:
                for eq_expr in eq_expressions:
                    if "=" in eq_expr:
                        lhs_part, rhs_part = eq_expr.split("=", 1)
                        lhs_part, rhs_part = lhs_part.strip(), rhs_part.strip()
                        if lhs_part not in all_values_for_verification:
                            try:
                                from physics_reasoning.solver.expression_parser import parse_expression
                                sym_expr = parse_expression(rhs_part)
                                res_val = float(sym_expr.evalf(subs={k: v for k, v in all_values_for_verification.items() if k in [str(s) for s in sym_expr.free_symbols]}))
                                all_values_for_verification[lhs_part] = res_val
                            except Exception:
                                pass

            v_res = self.verifier.verify(
                parsed_output=parsed_output,
                solve_result=solve_res,
                equations_used=equations_used,
                var_units=var_units,
                all_values=all_values_for_verification,
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
