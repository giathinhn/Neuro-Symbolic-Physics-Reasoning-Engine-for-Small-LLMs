"""Quantity extractor for mapping parsed LLM quantities to knowledge base definitions."""

from __future__ import annotations

import re

from physics_reasoning.core.enums import QuantityRole
from physics_reasoning.core.models import ParsedQuantity, PhysicsQuantity
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.units.unit_engine import UnitEngine

UNIT_TO_SYMBOL_MAP: dict[str, tuple[str, str, str]] = {
    # unit_key -> (default_name, default_symbol, standard_unit)
    "km/h": ("velocity", "v", "km/h"),
    "m/s": ("velocity", "v", "m/s"),
    "m/s^2": ("acceleration", "a", "m/s**2"),
    "m/s**2": ("acceleration", "a", "m/s**2"),
    "kg": ("mass", "m", "kg"),
    "g": ("mass", "m", "g"),
    "n/m^3": ("specific_weight", "d", "N/m**3"),
    "n/m**3": ("specific_weight", "d", "N/m**3"),
    "n": ("force", "F", "N"),
    "j": ("energy", "Q", "J"),
    "w": ("power", "P", "W"),
    "ohm": ("resistance", "R", "ohm"),
    "ω": ("resistance", "R", "ohm"),
    "v": ("voltage", "U", "V"),
    "a": ("current", "I", "A"),
    "°c": ("temperature", "t", "celsius"),
    "độ c": ("temperature", "t", "celsius"),
    "degc": ("temperature", "t", "celsius"),
    "k": ("temperature", "T", "kelvin"),
    "m^3": ("volume", "V", "m**3"),
    "m**3": ("volume", "V", "m**3"),
    "cm^3": ("volume", "V", "cm**3"),
    "m^2": ("area", "S", "m**2"),
    "m**2": ("area", "S", "m**2"),
    "cm^2": ("area", "S", "cm**2"),
    "m": ("displacement", "s", "m"),
    "cm": ("displacement", "s", "cm"),
    "mm": ("displacement", "s", "mm"),
    "km": ("displacement", "s", "km"),
    "s": ("time", "t", "s"),
    "giây": ("time", "t", "s"),
    "phút": ("time", "t", "minute"),
    "min": ("time", "t", "minute"),
    "giờ": ("time", "t", "hour"),
    "h": ("time", "t", "hour"),
}


class QuantityExtractor:
    """Extract and standardize physical quantities from parsed representations and text."""

    def __init__(self, knowledge_base: KnowledgeBase, unit_engine: UnitEngine | None = None):
        self.kb = knowledge_base
        self.unit_engine = unit_engine or UnitEngine()

    def extract_quantities_from_text(self, text: str) -> list[ParsedQuantity]:
        """Extract explicit physical quantities from problem text deterministically."""
        extracted: list[ParsedQuantity] = []
        found_values: set[float] = set()

        # Target detection from text
        text_lower = text.lower()
        if any(k in text_lower for k in ["nhiệt độ của nước khi cân bằng", "nhiệt độ cân bằng"]):
            extracted.append(
                ParsedQuantity(
                    name="temperature_equilibrium",
                    symbol="t_cb",
                    role=QuantityRole.TARGET,
                    unit="celsius",
                )
            )
        elif any(k in text_lower for k in ["lực đẩy ác-si-mét", "lực đẩy ac-si-met", "lực đẩy archimedes"]):
            extracted.append(
                ParsedQuantity(
                    name="buoyant_force",
                    symbol="F_A",
                    role=QuantityRole.TARGET,
                    unit="N",
                )
            )
        elif any(k in text_lower for k in ["điện trở tương đương"]):
            extracted.append(
                ParsedQuantity(
                    name="equivalent_resistance",
                    symbol="R_eq",
                    role=QuantityRole.TARGET,
                    unit="ohm",
                )
            )
        elif any(k in text_lower for k in ["công suất"]):
            extracted.append(
                ParsedQuantity(
                    name="power",
                    symbol="P",
                    role=QuantityRole.TARGET,
                    unit="W",
                )
            )
        elif any(k in text_lower for k in ["áp suất"]):
            extracted.append(
                ParsedQuantity(
                    name="pressure",
                    symbol="p",
                    role=QuantityRole.TARGET,
                    unit="Pa",
                )
            )
        elif any(k in text_lower for k in ["cường độ dòng điện"]):
            extracted.append(
                ParsedQuantity(
                    name="current",
                    symbol="I",
                    role=QuantityRole.TARGET,
                    unit="A",
                )
            )
        elif any(k in text_lower for k in ["quãng đường"]):
            extracted.append(
                ParsedQuantity(
                    name="path",
                    symbol="s",
                    role=QuantityRole.TARGET,
                    unit="m",
                )
            )
        elif any(k in text_lower for k in ["thời gian rơi"]):
            extracted.append(
                ParsedQuantity(
                    name="time",
                    symbol="t",
                    role=QuantityRole.TARGET,
                    unit="s",
                )
            )

        # Pattern 1: Explicit variable assignment e.g. R1 = 20 ohm, U = 24 V, g = 10 m/s^2, d = 10000 N/m^3
        var_assign_pattern = re.compile(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+(?:[.,][0-9]+)?)\s*([A-Za-z0-9°Ωμ/^*]+)?",
            re.IGNORECASE,
        )
        for m in var_assign_pattern.finditer(text):
            sym = m.group(1).strip()
            val_str = m.group(2).replace(",", ".").strip()
            unit_str = m.group(3).strip() if m.group(3) else None
            try:
                val = float(val_str)
                found_values.add(val)
                extracted.append(
                    ParsedQuantity(
                        name=sym,
                        symbol=sym,
                        value=val,
                        unit=unit_str,
                        role=QuantityRole.GIVEN,
                    )
                )
            except Exception:
                pass

        # Pattern 2: Contextual Number + Unit e.g. "200 g", "80 °C", "45 m", "15 km/h", "2 giờ"
        num_unit_pattern = re.compile(
            r"([0-9]+(?:[.,][0-9]+)?)\s*(km/h|m/s\^2|m/s\*\*2|m/s|kg|g|N/m\^3|N/m\*\*3|N|J|W|ohm|Ω|V|A|°C|độ C|degC|K|m\^3|m\*\*3|cm\^3|m\^2|m\*\*2|cm\^2|m|cm|mm|km|giây|s|phút|min|giờ|h)\b",
            re.IGNORECASE,
        )

        # Track occurrence count per physical type for indexing (m_1, m_2, t_1, t_2, etc.)
        type_counts: dict[str, int] = {}

        for m in num_unit_pattern.finditer(text):
            val_str = m.group(1).replace(",", ".").strip()
            raw_unit = m.group(2).strip().lower()
            start_pos = max(0, m.start() - 30)
            prefix_ctx = text[start_pos:m.start()].lower()

            try:
                val = float(val_str)
                if val not in found_values:
                    found_values.add(val)
                    if raw_unit in UNIT_TO_SYMBOL_MAP:
                        name, sym, std_unit = UNIT_TO_SYMBOL_MAP[raw_unit]

                        # Check special contextual mappings
                        if sym == "S":
                            if "tổng" in prefix_ctx or "tong" in prefix_ctx:
                                sym = "S"
                                name = "total_area"
                            elif "mỗi" in prefix_ctx or "moi" in prefix_ctx or "1 chân" in prefix_ctx:
                                sym = "S_1"
                                name = "single_area"
                        elif name in ("temperature", "mass"):
                            cnt = type_counts.get(name, 0) + 1
                            type_counts[name] = cnt
                            if cnt > 1 or any(k in text_lower for k in ["trộn", "tron", "pha"]):
                                sym = f"{sym}_{cnt}"
                                name = f"{name}_{cnt}"
                        elif ("trọng lượng" in prefix_ctx or "trong luong" in prefix_ctx) and raw_unit in ("n", "newton"):
                            name = "weight"
                            sym = "F"
                        elif name == "distance":
                            if "cao" in prefix_ctx:
                                sym = "h"
                                name = "height"
                            else:
                                sym = "s"
                                name = "distance"

                        extracted.append(
                            ParsedQuantity(
                                name=name,
                                symbol=sym,
                                value=val,
                                unit=std_unit,
                                role=QuantityRole.GIVEN,
                            )
                        )
            except Exception:
                pass

        return extracted

    def merge_with_text_quantities(
        self, parsed_quantities: list[ParsedQuantity], problem_text: str
    ) -> list[ParsedQuantity]:
        """Merge LLM-extracted quantities with deterministic text-extracted quantities."""
        existing_abs_values = {
            round(abs(float(q.value)), 4) for q in parsed_quantities if q.value is not None
        }
        existing_symbols = {q.symbol for q in parsed_quantities if q.symbol}
        has_target = any(q.role == QuantityRole.TARGET for q in parsed_quantities)

        text_quantities = self.extract_quantities_from_text(problem_text)
        merged = list(parsed_quantities)

        for tq in text_quantities:
            if tq.role == QuantityRole.TARGET:
                if not has_target:
                    merged.append(tq)
                    has_target = True
            elif tq.value is not None:
                val_rnd = round(abs(float(tq.value)), 4)
                if val_rnd not in existing_abs_values and tq.symbol not in existing_symbols:
                    existing_abs_values.add(val_rnd)
                    existing_symbols.add(tq.symbol)
                    merged.append(tq)

        return merged

    def standardize_quantity(self, parsed: ParsedQuantity) -> PhysicsQuantity:
        """Enrich a ParsedQuantity with knowledge base metadata (dimension, si_unit)."""
        kb_def = self.kb.get_quantity_by_name(parsed.name) or self.kb.get_quantity_by_symbol(parsed.symbol)

        dimension = ""
        si_unit = ""
        aliases = []

        if kb_def:
            dimension = kb_def.dimension
            si_unit = kb_def.si_unit
            aliases = kb_def.aliases

        # If dimension missing, infer from unit if provided
        if not dimension and parsed.unit:
            dimension = self.unit_engine.get_dimension(parsed.unit)

        # If si_unit missing, infer from unit if provided
        if not si_unit and parsed.unit:
            try:
                _, actual_si_unit = self.unit_engine.to_si(1.0, parsed.unit)
                si_unit = actual_si_unit
            except Exception:
                si_unit = parsed.unit

        return PhysicsQuantity(
            name=kb_def.name if kb_def else parsed.name,
            symbol=parsed.symbol,
            value=parsed.value,
            unit=parsed.unit,
            dimension=dimension,
            si_unit=si_unit or (parsed.unit or ""),
            is_target=(parsed.role == QuantityRole.TARGET),
            is_given=(parsed.role == QuantityRole.GIVEN),
            aliases=aliases,
        )

    def standardize_all(self, parsed_quantities: list[ParsedQuantity]) -> list[PhysicsQuantity]:
        """Standardize a list of parsed quantities."""
        return [self.standardize_quantity(q) for q in parsed_quantities]
