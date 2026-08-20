"""Knowledge Base manager for qualitative physics principles, phenomena, and misconceptions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from physics_reasoning.core.exceptions import KnowledgeBaseError
from physics_reasoning.core.models import QualitativePrinciple


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
            "những", "một", "nào", "hãy", "gì", "như", "thế", "này", "why", "how", "what", "is", "the", "a", "an"
        }
        q_words = set(re.findall(r"\w+", q_lower)) - stop_words

        scored: list[tuple[float, QualitativePrinciple]] = []

        for p in self.principles.values():
            score = 0.0

            # Check keywords
            for kw in p.keywords:
                kw_lower = kw.lower()
                if kw_lower in q_lower:
                    score += 10.0

            # Check typical phenomena similarity
            for tp in p.typical_phenomena:
                tp_words = set(re.findall(r"\w+", tp.lower())) - stop_words
                overlap = len(tp_words.intersection(q_words))
                if overlap >= 2:
                    score += overlap * 4.0
                elif overlap == 1:
                    score += 2.0

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
