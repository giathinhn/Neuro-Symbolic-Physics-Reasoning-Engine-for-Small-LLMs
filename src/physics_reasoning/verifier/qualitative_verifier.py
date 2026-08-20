"""Verification engine for qualitative physics phenomenon explanations."""

from __future__ import annotations

from datetime import datetime

from physics_reasoning.core.enums import ErrorSeverity, ErrorType
from physics_reasoning.core.models import (
    QualitativeParsedOutput,
    VerificationError,
    VerificationResult,
)
from physics_reasoning.physics.qualitative_kb import QualitativeKnowledgeBase


class PrincipleRelevanceCheck:
    """Check that cited qualitative principles are relevant and exist in knowledge base."""

    def __init__(self, kb: QualitativeKnowledgeBase):
        self.kb = kb

    @property
    def name(self) -> str:
        return "principle_relevance"

    def run(
        self,
        problem_text: str,
        output: QualitativeParsedOutput,
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        if not output.core_principles:
            errors.append(
                VerificationError(
                    error_type=ErrorType.INVALID_EQUATION,
                    severity=ErrorSeverity.ERROR,
                    message="Không có nguyên lý hay định luật vật lý nào được chỉ định làm cơ sở giải thích.",
                    suggestion="Hãy xác định định luật/nguyên lý cốt lõi (ví dụ: Quán tính, Áp suất p = F/S, Lực ma sát, Sự nở vì nhiệt...).",
                )
            )
            return errors

        # Check whether cited principle exists in KB or is semantically matched
        relevant_principles = self.kb.find_relevant_principles(problem_text, top_k=3)
        relevant_ids = {p.id for p in relevant_principles}
        relevant_names = {p.name.lower() for p in relevant_principles}

        cited = output.core_principles
        # If none of the cited principles match or overlap with domain
        has_valid_citation = False
        for c in cited:
            c_clean = c.lower().strip()
            if c in self.kb.principles:
                has_valid_citation = True
                break
            if any(c_clean in r_name for r_name in relevant_names) or any(c in relevant_ids for c in [c_clean]):
                has_valid_citation = True
                break

        if not has_valid_citation and relevant_principles:
            expected = ", ".join(f"'{p.name}' ({p.id})" for p in relevant_principles[:2])
            errors.append(
                VerificationError(
                    error_type=ErrorType.INVALID_EQUATION,
                    severity=ErrorSeverity.WARNING,
                    message=f"Nguyên lý viện dẫn '{cited}' có thể chưa phù hợp nhất cho hiện tượng này.",
                    suggestion=f"Xem xét áp dụng nguyên lý: {expected}",
                )
            )

        return errors


class CausalCompletenessCheck:
    """Check that the causal chain is continuous and logically complete."""

    @property
    def name(self) -> str:
        return "causal_completeness"

    def run(
        self,
        problem_text: str,
        output: QualitativeParsedOutput,
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        if len(output.causal_chain) < 2:
            errors.append(
                VerificationError(
                    error_type=ErrorType.SUBSTITUTION_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="Chuỗi giải thích nhân quả quá ngắn (cần ít nhất 2 bước: Tác động ban đầu -> Cơ chế vật lý -> Kết quả).",
                    suggestion="Hãy phân tích rõ: (1) Trạng thái ban đầu và tác động, (2) Cơ chế vật lý xảy ra theo định luật, (3) Hiện tượng quan sát được.",
                )
            )
            return errors

        # Check if mechanism descriptions are substantive
        has_substantive_mechanism = any(
            len(step.physical_mechanism.strip()) >= 15
            for step in output.causal_chain
        )
        if not has_substantive_mechanism:
            errors.append(
                VerificationError(
                    error_type=ErrorType.SUBSTITUTION_ERROR,
                    severity=ErrorSeverity.WARNING,
                    message="Các bước giải thích chưa nêu rõ cơ chế vật lý chi tiết.",
                    suggestion="Cần giải thích rõ tại sao hiện tượng lại xảy ra dựa trên bản chất định luật vật lý.",
                )
            )

        return errors


class MisconceptionCheck:
    """Check against known physics misconceptions and anti-patterns."""

    def __init__(self, kb: QualitativeKnowledgeBase):
        self.kb = kb

    @property
    def name(self) -> str:
        return "misconception_check"

    def run(
        self,
        problem_text: str,
        output: QualitativeParsedOutput,
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        # Combine all explanation text
        full_text = output.conclusion + " " + " ".join(
            f"{s.state_or_action} {s.physical_mechanism}" for s in output.causal_chain
        )

        detected = self.kb.scan_for_misconceptions(full_text)
        for d in detected:
            errors.append(
                VerificationError(
                    error_type=ErrorType.IMPOSSIBLE_VALUE,
                    severity=ErrorSeverity.ERROR,
                    message=f"Phát hiện ngộ nhận vật lý: '{d['name']}'. Lỗi: {d['explanation']}",
                    suggestion=d.get("correction"),
                    context={"misconception_id": d["id"]},
                )
            )

        return errors


class ScientificTerminologyCheck:
    """Check that conclusion is unambiguous and uses precise scientific terms."""

    @property
    def name(self) -> str:
        return "scientific_terminology"

    def run(
        self,
        problem_text: str,
        output: QualitativeParsedOutput,
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []

        if not output.conclusion or len(output.conclusion.strip()) < 10:
            errors.append(
                VerificationError(
                    error_type=ErrorType.MISSING_QUANTITY,
                    severity=ErrorSeverity.ERROR,
                    message="Lời kết luận tóm tắt hiện tượng chưa hoàn chỉnh hoặc quá ngắn.",
                    suggestion="Cung cấp câu kết luận rõ ràng, trực tiếp trả lời câu hỏi của bài toán.",
                )
            )

        return errors


class QualitativeVerificationPipeline:
    """Orchestrate all qualitative verification checks."""

    def __init__(self, qualitative_kb: QualitativeKnowledgeBase | None = None):
        self.kb = qualitative_kb or QualitativeKnowledgeBase()
        if not self.kb.principles:
            try:
                self.kb.load()
            except Exception:
                pass

        self.checks = [
            PrincipleRelevanceCheck(self.kb),
            CausalCompletenessCheck(),
            MisconceptionCheck(self.kb),
            ScientificTerminologyCheck(),
        ]

    def verify(
        self,
        problem_text: str,
        parsed_output: QualitativeParsedOutput,
    ) -> VerificationResult:
        """Run all verification checks on qualitative output."""
        all_errors: list[VerificationError] = []
        checks_performed: list[str] = []
        checks_passed: list[str] = []

        for check in self.checks:
            checks_performed.append(check.name)
            errs = check.run(problem_text, parsed_output)
            if errs:
                all_errors.extend(errs)
            else:
                checks_passed.append(check.name)

        has_blocking_errors = any(
            e.severity in (ErrorSeverity.ERROR, ErrorSeverity.FATAL) for e in all_errors
        )
        is_valid = not has_blocking_errors

        confidence = (
            len(checks_passed) / len(checks_performed)
            if checks_performed and not has_blocking_errors
            else 0.0
        )

        return VerificationResult(
            is_valid=is_valid,
            errors=all_errors,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            confidence=confidence,
            timestamp=datetime.now(),
        )
