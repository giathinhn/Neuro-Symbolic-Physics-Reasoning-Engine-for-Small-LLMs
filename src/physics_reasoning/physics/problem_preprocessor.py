"""Problem text preprocessor and noise reducer for physics reasoning."""

from __future__ import annotations

import re


# Distractor / boundary condition boilerplate patterns
BOILERPLATE_PATTERNS = [
    # Vietnamese thermal boundary conditions
    re.compile(
        r"(?:Bỏ qua|Không kể|Bỏ qua sự|Không tính)\s+(?:mất mát|thất thoát|hao phí|truyền|trao đổi)\s+nhiệt\s*(?:ra|cho|với)?\s*(?:môi trường|bình chứa|nhiệt lượng kế|bình|xung quanh)?(?:\s+và\s+môi trường)?(?:\s+xung quanh)?[.,;]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Bỏ qua|Không tính)\s+(?:sự\s+)?hấp thụ nhiệt của (?:bình chứa|nhiệt lượng kế|môi trường)[.,;]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Bỏ qua|Không tính)\s+sự (?:giãn nở|co giãn) vì nhiệt (?:của bình|của vật)?[.,;]?",
        re.IGNORECASE,
    ),
    # Vietnamese mechanics boundary conditions
    re.compile(
        r"(?:Bỏ qua|Không tính|Bỏ qua mọi)\s+(?:ma sát|lực ma sát|lực cản|sức cản|sức cản của không khí|lực cản của không khí|lực cản của môi trường)[.,;]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Bỏ qua|Không tính)\s+(?:khối lượng|trọng lượng)\s+(?:của\s+)?(?:dây|dây nối|ròng rọc|dây và ròng rọc|cần trục|thước)[.,;]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Coi|Giả sử)\s+(?:dây không co dãn|dây không dãn|dây nhẹ|ròng rọc lý tưởng|không có sự mất mát năng lượng)[.,;]?",
        re.IGNORECASE,
    ),
    # English boundary conditions
    re.compile(
        r"(?:(?:Ignoring|Neglecting|Assume no|Without|Neglect)\s+(?:heat loss|thermal loss|energy loss|friction|air resistance|heat transfer)(?:\s+to\s+the\s+surroundings|\s+to\s+the\s+container|\s+to\s+ambient)?)[.,;]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:Assume|Consider)\s+(?:massless\s+and\s+frictionless\s+pulley|massless\s+pulley|frictionless\s+surface|ideal\s+gas|inextensible\s+string|ideal\s+conditions))[.,;]?",
        re.IGNORECASE,
    ),
]


class ProblemPreprocessor:
    """Preprocess and clean physics problem statements to remove distractors and noise."""

    @classmethod
    def denoise_text(cls, text: str) -> tuple[str, list[str]]:
        """Extract boundary condition clauses and produce a clean core problem statement.

        Args:
            text: Raw input problem text.

        Returns:
            tuple of (clean_text, extracted_conditions)
        """
        if not text:
            return "", []

        cleaned = text.strip()
        conditions: list[str] = []

        for pat in BOILERPLATE_PATTERNS:
            for match in pat.finditer(cleaned):
                matched_str = match.group(0).strip(" .,;")
                if matched_str:
                    conditions.append(matched_str)
            cleaned = pat.sub(" ", cleaned)

        # Normalize plus signs used as connectors e.g. "200g ở 80°C + 300g ở 20°C" -> "200g ở 80°C và 300g ở 20°C"
        cleaned = re.sub(r"\s*\+\s*", " và ", cleaned)

        # Clean multiple spaces and whitespace before punctuation
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([.,?!])", r"\1", cleaned)
        cleaned = re.sub(r"\.{2,}", ".", cleaned)
        cleaned = cleaned.strip()

        return cleaned, conditions
