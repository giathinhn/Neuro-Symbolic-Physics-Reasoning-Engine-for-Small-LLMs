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

    @classmethod
    def extract_implicit_physical_constants(cls, text: str) -> dict[str, tuple[float, str]]:
        """Extract implied physical quantities and boundary constants from problem phrasing.

        Examples:
        - "đun sôi nước từ 20°C" -> t_final = 100°C (t2 = 100°C)
        - "thả rơi tự do" / "từ trạng thái nghỉ" -> v0 = 0 m/s
        - "dừng hẳn" -> v_final = 0 m/s
        - "đá đang tan" -> t1 = 0°C
        - "lấy g = 10" or gravity problems -> g = 10 m/s^2

        Returns:
            dict mapping variable name -> (value, unit)
        """
        constants: dict[str, tuple[float, str]] = {}
        t_lower = text.lower()

        # 1. Boiling water (Nhiệt độ sôi của nước = 100°C)
        if any(w in t_lower for w in ["đun sôi", "nước sôi", "sôi", "boiling", "boil"]):
            constants["t_boil"] = (100.0, "°C")
            constants["t2"] = (100.0, "°C")
            constants["t_final"] = (100.0, "°C")
            constants["T_final"] = (100.0, "°C")
            constants["T2"] = (100.0, "°C")

        # 2. Melting ice (Nước đá đang tan = 0°C)
        if any(w in t_lower for w in ["đá đang tan", "nước đá", "melting ice"]):
            constants["t_ice"] = (0.0, "°C")
            constants["t1"] = (0.0, "°C")
            constants["t_initial"] = (0.0, "°C")

        # 3. Starting from rest / Free fall (Vận tốc ban đầu = 0 m/s)
        if any(w in t_lower for w in ["rơi tự do", "thả rơi", "không vận tốc đầu", "từ trạng thái nghỉ", "đang đứng yên", "from rest"]):
            constants["v0"] = (0.0, "m/s")
            constants["v_0"] = (0.0, "m/s")
            constants["v_i"] = (0.0, "m/s")
            constants["v_initial"] = (0.0, "m/s")

        # 4. Braking until full stop (Vận tốc cuối = 0 m/s)
        if any(w in t_lower for w in ["dừng hẳn", "dừng lại", "hãm phanh đến khi dừng", "comes to rest", "comes to stop"]):
            constants["v"] = (0.0, "m/s")
            constants["v_f"] = (0.0, "m/s")
            constants["v_final"] = (0.0, "m/s")

        # 5. Gravity acceleration g
        g_match = re.search(r"\bg\s*=\s*(\d+(?:\.\d+)?)\s*(?:m/s\^2|m/s\*\*2|m/s2)?", t_lower)
        if g_match:
            constants["g"] = (float(g_match.group(1)), "m/s^2")
        elif any(w in t_lower for w in ["trọng lượng", "rơi tự do", "thế năng", "nâng", "cần trục", "máy tời", "áp suất chất lỏng", "lực đẩy ác-si-mét", "lực đẩy archimedes", "lực kế"]):
            constants["g"] = (10.0, "m/s^2")

        # 6. Specific heat / density of water when mentioned generically
        if "nước" in t_lower or "water" in t_lower:
            if "c_water" not in constants and any(w in t_lower for w in ["nhiệt", "đun", "nóng", "nguội", "pha"]):
                constants["c_water"] = (4200.0, "J/(kg.K)")
                constants["c"] = (4200.0, "J/(kg.K)")
            if any(w in t_lower for w in ["áp suất", "chìm", "nổi", "lực đẩy", "thể tích"]):
                if "d_water" not in constants:
                    constants["d_water"] = (10000.0, "N/m^3")
                if "rho_water" not in constants:
                    constants["rho_water"] = (1000.0, "kg/m^3")

        return constants
