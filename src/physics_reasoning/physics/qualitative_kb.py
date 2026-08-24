"""Knowledge Base manager for qualitative physics principles, phenomena, and misconceptions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from physics_reasoning.core.exceptions import KnowledgeBaseError
from physics_reasoning.core.models import (
    CausalStep,
    QualitativeParsedOutput,
    QualitativePrinciple,
)


class QualitativeKnowledgeBase:
    """In-memory qualitative physics knowledge base."""

    def __init__(self, qualitative_dir: str | Path = "data/knowledge/qualitative"):
        self.qualitative_dir = Path(qualitative_dir)
        self.principles: dict[str, QualitativePrinciple] = {}  # id -> QualitativePrinciple
        self.misconceptions: list[dict[str, Any]] = []

    def load(self) -> None:
        """Load qualitative principles and misconception rules from YAML."""
        if not self.qualitative_dir.exists():
            raise KnowledgeBaseError(
                f"Qualitative knowledge directory not found: {self.qualitative_dir}"
            )

        # 1. Load principles
        p_file = self.qualitative_dir / "principles.yaml"
        if p_file.exists():
            try:
                with open(p_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    for item in data.get("principles", []):
                        principle = QualitativePrinciple(**item)
                        self.principles[principle.id] = principle
            except Exception as e:
                raise KnowledgeBaseError(f"Failed to load qualitative principles: {e}") from e

        # 2. Load misconceptions
        m_file = self.qualitative_dir / "misconceptions.yaml"
        if m_file.exists():
            try:
                with open(m_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    self.misconceptions = data.get("misconceptions", [])
            except Exception as e:
                raise KnowledgeBaseError(f"Failed to load misconceptions: {e}") from e

    def get_principle(self, principle_id: str) -> QualitativePrinciple | None:
        """Get qualitative principle by ID."""
        return self.principles.get(principle_id)

    def find_relevant_principles(self, query_text: str, top_k: int = 3) -> list[QualitativePrinciple]:
        """Find most relevant qualitative principles matching the query text.

        Ranks by matching keywords, typical phenomena, and domain tokens.
        """
        q_lower = query_text.lower()
        stop_words = {
            "tại", "sao", "khi", "thì", "lại", "bị", "có", "là", "và", "trong",
            "cho", "của", "được", "ra", "vào", "với", "hơn", "so", "ở", "các",
            "những", "một", "nào", "hãy", "gì", "như", "thế", "này", "why", "how", "what", "is", "the", "a", "an",
            "giải", "thích", "vì", "sao", "hiện", "tượng"
        }
        q_words = set(re.findall(r"\w+", q_lower)) - stop_words

        scored: list[tuple[float, QualitativePrinciple]] = []

        for p in self.principles.values():
            score = 0.0

            # 1. Exact keyword match
            for kw in p.keywords:
                kw_lower = kw.lower()
                if kw_lower in q_lower:
                    # Longer keywords get higher weights
                    score += 10.0 + len(kw_lower.split()) * 3.0

            # 2. Typical phenomena similarity (n-gram & token overlap)
            for tp in p.typical_phenomena:
                tp_lower = tp.lower()
                # Direct subphrase match
                if any(phrase in q_lower for phrase in [tp_lower[:20], tp_lower[-20:]] if len(phrase) >= 10):
                    score += 25.0

                tp_words = set(re.findall(r"\w+", tp_lower)) - stop_words
                overlap = len(tp_words.intersection(q_words))
                if overlap >= 3:
                    score += overlap * 6.0
                elif overlap == 2:
                    score += 8.0
                elif overlap == 1:
                    score += 2.0

            # 3. Principle name match
            if p.name.lower() in q_lower:
                score += 20.0

            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def scan_for_misconceptions(self, text: str) -> list[dict[str, Any]]:
        """Scan explanation text against known physics misconception patterns.

        Returns list of matched misconception warnings.
        """
        t_lower = text.lower()
        matched: list[dict[str, Any]] = []

        for m in self.misconceptions:
            patterns = m.get("pattern_regex", [])
            for pat in patterns:
                if re.search(pat, t_lower, flags=re.IGNORECASE):
                    matched.append(
                        {
                            "id": m.get("id"),
                            "name": m.get("name"),
                            "explanation": m.get("explanation"),
                            "correction": m.get("correction"),
                            "matched_pattern": pat,
                        }
                    )
                    break

        return matched

    def synthesize_fallback_explanation(
        self, problem_text: str, principle: QualitativePrinciple
    ) -> QualitativeParsedOutput:
        """Synthesize a canonical qualitative explanation when LLM verification fails."""
        # Find best matching typical phenomenon if any
        q_lower = problem_text.lower()
        matched_tp = ""
        for tp in principle.typical_phenomena:
            tp_words = set(re.findall(r"\w+", tp.lower()))
            q_words = set(re.findall(r"\w+", q_lower))
            if len(tp_words.intersection(q_words)) >= 2:
                matched_tp = tp
                break

        steps: list[CausalStep] = [
            CausalStep(
                step_number=1,
                state_or_action=f"Xem xét trạng thái ban đầu và hiện tượng: '{problem_text.strip()}'.",
                physical_mechanism=f"Hiện tượng này chịu sự chi phối trực tiếp của {principle.name}.",
                governing_principle=principle.id,
            ),
            CausalStep(
                step_number=2,
                state_or_action="Quá trình biến đổi vật lý diễn ra theo quy luật tự nhiên.",
                physical_mechanism=principle.description,
                governing_principle=principle.id,
            ),
            CausalStep(
                step_number=3,
                state_or_action="Kết quả quan sát được trong thực tế.",
                physical_mechanism=(
                    matched_tp
                    if matched_tp
                    else f"Theo {principle.name}, điều này dẫn tới hiện tượng được nêu trong bài toán."
                ),
                governing_principle=principle.id,
            ),
        ]

        conclusion = (
            matched_tp
            if matched_tp
            else f"Hiện tượng '{problem_text.strip()}' xảy ra là do {principle.name}: {principle.description}"
        )

        return QualitativeParsedOutput(
            problem_understanding=f"Giải thích hiện tượng: {problem_text.strip()}",
            observed_phenomenon=problem_text.strip(),
            core_principles=[principle.id],
            causal_chain=steps,
            conclusion=conclusion,
            scientific_keywords=principle.keywords[:5],
        )
