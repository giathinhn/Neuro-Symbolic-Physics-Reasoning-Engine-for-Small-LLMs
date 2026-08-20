"""Problem type classifier distinguishing quantitative vs qualitative physics problems."""

from __future__ import annotations

import re

from physics_reasoning.core.enums import ProblemType

QUALITATIVE_PATTERNS = [
    r"\btại sao\b",
    r"\bvì sao\b",
    r"\bgiải thích\b",
    r"\bhiện tượng gì\b",
    r"\bnguyên nhân\b",
    r"\bnhư thế nào\b",
    r"\bnhận xét\b",
    r"\bwhy\b",
    r"\bexplain\b",
    r"\bhow does\b",
    r"\bwhat causes\b",
    r"\bwhat happens when\b",
    r"\bdescribe the phenomenon\b",
]

QUANTITATIVE_PATTERNS = [
    r"\btính\b",
    r"\btính toán\b",
    r"\bbằng bao nhiêu\b",
    r"\bxác định độ lớn\b",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bdetermine the value\b",
    r"\bfind the magnitude\b",
]


class ProblemClassifier:
    """Classify physics problems into quantitative vs qualitative types."""

    @staticmethod
    def classify(problem_text: str) -> ProblemType:
        """Classify problem text.

        Returns:
            ProblemType.QUALITATIVE or ProblemType.QUANTITATIVE
        """
        text_lower = problem_text.lower().strip()

        # Check qualitative indicators
        qual_score = 0
        for pat in QUALITATIVE_PATTERNS:
            if re.search(pat, text_lower):
                qual_score += 2

        # Check quantitative indicators
        quant_score = 0
        for pat in QUANTITATIVE_PATTERNS:
            if re.search(pat, text_lower):
                quant_score += 2

        # Count numbers with physical units
        numeric_unit_matches = re.findall(
            r"\d+(?:\.\d+)?\s*(?:kg|g|m/s|km/h|m/s\^2|m/s\*\*2|n|j|w|pa|m|km|s|min|h|cm|dm|lit|l|ml)",
            text_lower,
        )
        if len(numeric_unit_matches) >= 2:
            quant_score += len(numeric_unit_matches) * 1.5

        if qual_score > quant_score:
            return ProblemType.QUALITATIVE
        elif quant_score > qual_score:
            return ProblemType.QUANTITATIVE
        else:
            # Fallback: if 'why' / 'tại sao' present anywhere, treat as qualitative
            if any(q in text_lower for q in ["tại sao", "vì sao", "giải thích", "why", "explain"]):
                return ProblemType.QUALITATIVE
            # If numbers are present, default to quantitative
            if re.search(r"\d+", text_lower):
                return ProblemType.QUANTITATIVE
            return ProblemType.QUALITATIVE
