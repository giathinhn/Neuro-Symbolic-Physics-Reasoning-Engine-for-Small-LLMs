"""Quantity extractor for mapping parsed LLM quantities to knowledge base definitions."""

from __future__ import annotations

from physics_reasoning.core.enums import QuantityRole
from physics_reasoning.core.models import ParsedQuantity, PhysicsQuantity
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.units.unit_engine import UnitEngine


class QuantityExtractor:
    """Extract and standardize physical quantities from parsed representations."""

    def __init__(self, knowledge_base: KnowledgeBase, unit_engine: UnitEngine | None = None):
        self.kb = knowledge_base
        self.unit_engine = unit_engine or UnitEngine()

    def standardize_quantity(self, parsed: ParsedQuantity) -> PhysicsQuantity:
        """Enrich a ParsedQuantity with knowledge base metadata (dimension, si_unit)."""
        # Look up definition in knowledge base by name or symbol
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
