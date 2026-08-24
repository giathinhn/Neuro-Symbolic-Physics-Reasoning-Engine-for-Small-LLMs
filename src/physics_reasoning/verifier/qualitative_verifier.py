"""Verification engine for qualitative physics phenomenon explanations."""

from __future__ import annotations

import re
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
                    suggestion="Hãy xác định định luật/nguyên lý cốt lõi (ví dụ: Quán tính, Áp suất p = F/S, Lực ma sát, Nhiễm điện do cọ xát, Sự ngưng tụ...).",
                )
            )
            return errors

        # Check whether cited principle matches the relevant principles for this problem
        relevant_principles = self.kb.find_relevant_principles(problem_text, top_k=3)
        relevant_ids = {p.id for p in relevant_principles}
        relevant_names = {p.name.lower() for p in relevant_principles}

        cited = output.core_principles
        has_valid_citation = False
        for c in cited:
            c_clean = c.lower().strip()
            if c_clean in relevant_ids or any(c_clean in r_name for r_name in relevant_names) or any(r_id in c_clean for r_id in relevant_ids):
                has_valid_citation = True
                break

        if not has_valid_citation and relevant_principles:
            expected = ", ".join(f"'{p.name}' ({p.id})" for p in relevant_principles[:2])
            errors.append(
                VerificationError(
                    error_type=ErrorType.INVALID_EQUATION,
                    severity=ErrorSeverity.ERROR,
                    message=f"Nguyên lý viện dẫn '{cited}' không chính xác cho hiện tượng này.",
                    suggestion=f"Hiện tượng này cần áp dụng nguyên lý: {expected}",
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
                    message="Chuỗi giải thích nhân quả quá ngắn (cần ít nhất 2 bước: Trạng thái/Tác động ban đầu -> Cơ chế vật lý -> Kết quả).",
                    suggestion="Hãy phân tích rõ: (1) Trạng thái ban đầu và tác động, (2) Cơ chế vật lý xảy ra theo định luật, (3) Hiện tượng quan sát được.",
                )
            )
            return errors

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


class CausalDirectionalityCheck:
    """Check physical directionality and causal correctness of physical claims."""

    @property
    def name(self) -> str:
        return "causal_directionality"

    def run(
        self,
        problem_text: str,
        output: QualitativeParsedOutput,
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []
        p_lower = problem_text.lower()
        full_text = (
            output.conclusion + " " + " ".join(f"{s.state_or_action} {s.physical_mechanism}" for s in output.causal_chain)
        ).lower()

        # 1. Inertia braking: Must lean forward (chúi/ngả về phía trước), NOT backward (về phía sau)
        if ("phanh gấp" in p_lower or "thắng gấp" in p_lower or "dừng đột ngột" in p_lower) and ("người" in p_lower or "hành khách" in p_lower):
            if re.search(r"(ngả|nghiêng|chúi|ngã)\s*(ra|về)?\s*(phía\s+)?sau", full_text):
                errors.append(
                    VerificationError(
                        error_type=ErrorType.IMPOSSIBLE_VALUE,
                        severity=ErrorSeverity.ERROR,
                        message="Sai lệch hướng quán tính: Khi xe phanh gấp, người phải ngã chúi về PHÍA TRƯỚC (không phải ngả về phía sau).",
                        suggestion="Do có quán tính, thân người tiếp tục duy trì vận tốc về phía trước trong khi xe đã dừng lại, dẫn tới ngã chúi về phía trước.",
                    )
                )

        # 2. Inertia turning: Turning right -> leans left; turning left -> leans right
        if "rẽ phải" in p_lower:
            if re.search(r"(ngả|nghiêng|chúi|ngã)\s*(sang|về)?\s*(phía\s+)?phải", full_text):
                errors.append(
                    VerificationError(
                        error_type=ErrorType.IMPOSSIBLE_VALUE,
                        severity=ErrorSeverity.ERROR,
                        message="Sai lệch hướng quán tính: Khi xe rẽ phải, người phải nghiêng sang BÊN TRÁI (theo hướng chuyển động cũ).",
                        suggestion="Giải thích rằng do quán tính, người tiếp tục duy trì hướng chuyển động thẳng ban đầu nên có xu hướng nghiêng sang trái khi xe ngoặt sang phải.",
                    )
                )
        if "rẽ trái" in p_lower:
            if re.search(r"(ngả|nghiêng|chúi|ngã)\s*(sang|về)?\s*(phía\s+)?trái", full_text):
                errors.append(
                    VerificationError(
                        error_type=ErrorType.IMPOSSIBLE_VALUE,
                        severity=ErrorSeverity.ERROR,
                        message="Sai lệch hướng quán tính: Khi xe rẽ trái, người phải nghiêng sang BÊN PHẢI.",
                        suggestion="Giải thích theo quán tính duy trì chuyển động thẳng cũ.",
                    )
                )

        # 3. Refraction depth: Pool/lake bottom looks shallower (nông hơn), NOT deeper (sâu hơn)
        if ("đáy hồ" in p_lower or "đáy bể" in p_lower or "bể bơi" in p_lower) and "nông" in p_lower:
            if re.search(r"thấy.*sâu hơn", full_text):
                errors.append(
                    VerificationError(
                        error_type=ErrorType.IMPOSSIBLE_VALUE,
                        severity=ErrorSeverity.ERROR,
                        message="Sai lệch khúc xạ ánh sáng: Đáy hồ/bể bơi khi nhìn từ trên xuống phải trông NÔNG HƠN thực tế.",
                        suggestion="Khúc xạ ánh sáng làm ảnh ảo của đáy bể bị nâng cao lên, tạo cảm giác bể nông hơn.",
                    )
                )

        # 4. Communicating vessels (Bình thông nhau & Vòi ấm)
        if any("communicating_vessels" in p for p in output.core_principles) and ("vòi" in p_lower or "ấm" in p_lower):
            if "tràn" not in full_text and "chảy ra ngoài" not in full_text:
                errors.append(
                    VerificationError(
                        error_type=ErrorType.SUBSTITUTION_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message="Thiếu cơ chế bình thông nhau: Cần giải thích mực nước ở ấm và vòi luôn ngang nhau để tránh bị tràn nước ra ngoài khi đổ đầy.",
                        suggestion="Nêu rõ theo nguyên lý bình thông nhau, các mực nước trong ấm và vòi luôn ở cùng một độ cao (mặt thoáng ngang nhau). Nếu vòi thấp hơn miệng ấm thì khi đổ đầy nước sẽ bị tràn ra ngoài vòi.",
                    )
                )

        # 5. Electrostatic charging (Cánh quạt bám bụi)
        if any("electrostatic" in p for p in output.core_principles):
            if not any(k in full_text for k in ["nhiễm điện", "tích điện", "tĩnh điện"]):
                errors.append(
                    VerificationError(
                        error_type=ErrorType.SUBSTITUTION_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message="Thiếu cơ chế tĩnh điện: Cần chỉ rõ cánh quạt quay cọ xát với không khí bị nhiễm điện nên có khả năng hút các hạt bụi nhỏ nhẹ.",
                        suggestion="Giải thích rằng khi quay, cánh quạt cọ xát mạnh với không khí và bị nhiễm điện (tích điện), từ đó sinh lực hút tĩnh điện giữ các hạt bụi nhẹ bám vào.",
                    )
                )

        return errors


class AntiTautologyCheck:
    """Detect circular trivial explanations and non-answers."""

    @property
    def name(self) -> str:
        return "anti_tautology"

    def run(
        self,
        problem_text: str,
        output: QualitativeParsedOutput,
    ) -> list[VerificationError]:
        errors: list[VerificationError] = []
        conc_lower = output.conclusion.lower().strip()

        # Tautological phrases that avoid explaining mechanism
        tautology_patterns = [
            r"vì hiện tượng này là như vậy",
            r"do tính chất tự nhiên của nó",
            r"vì nó vốn dĩ như thế",
            r"vì vật làm như vậy thì nó như vậy",
        ]
        for pat in tautology_patterns:
            if re.search(pat, conc_lower):
                errors.append(
                    VerificationError(
                        error_type=ErrorType.SUBSTITUTION_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message="Lời giải thích mang tính lặp luận vòng vo (tautology), chưa chỉ ra cơ chế định luật vật lý thực sự.",
                        suggestion="Hãy dựa vào nguyên lý vật lý (ví dụ: cọ xát gây nhiễm điện, dãn nở vì nhiệt không đều, định luật phản xạ...) để giải thích từng bước.",
                    )
                )
                break

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

        if not output.conclusion or len(output.conclusion.strip()) < 15:
            errors.append(
                VerificationError(
                    error_type=ErrorType.MISSING_QUANTITY,
                    severity=ErrorSeverity.ERROR,
                    message="Lời kết luận tóm tắt hiện tượng chưa hoàn chỉnh hoặc quá ngắn (dưới 15 ký tự).",
                    suggestion="Cung cấp câu kết luận khoa học rõ ràng, trực tiếp trả lời câu hỏi của bài toán.",
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
            CausalDirectionalityCheck(),
            AntiTautologyCheck(),
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
