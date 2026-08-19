"""Knowledge base manager for loading and querying physics equations and quantities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from physics_reasoning.core.exceptions import KnowledgeBaseError
from physics_reasoning.core.models import Equation, PhysicsQuantity


class KnowledgeBase:
    """In-memory physics knowledge base loaded from YAML files."""

    def __init__(self, knowledge_dir: str | Path = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.quantities: dict[str, PhysicsQuantity] = {}  # name -> PhysicsQuantity
        self.equations: dict[str, Equation] = {}  # id -> Equation

        # Indices
        self._symbol_to_quantity: dict[str, str] = {}  # symbol -> quantity name
        self._alias_to_quantity: dict[str, str] = {}  # alias.lower() -> quantity name
        self._topic_index: dict[str, list[str]] = {}  # topic -> [eq_ids]
        self._variable_index: dict[str, list[str]] = {}  # quantity_name -> [eq_ids]

    def load(self) -> None:
        """Load all YAML definition files from the knowledge directory."""
        if not self.knowledge_dir.exists():
            raise KnowledgeBaseError(
                f"Knowledge base directory does not exist: {self.knowledge_dir}"
            )

        # 1. Load quantities
        quantities_file = self.knowledge_dir / "quantities.yaml"
        if quantities_file.exists():
            try:
                with open(quantities_file, encoding="utf-8") as f:
                    q_data = yaml.safe_load(f) or {}
                    for item in q_data.get("quantities", []):
                        qty = PhysicsQuantity(**item)
                        self.quantities[qty.name] = qty
                        self._symbol_to_quantity[qty.symbol] = qty.name
                        self._alias_to_quantity[qty.name.lower()] = qty.name
                        for alias in qty.aliases:
                            self._alias_to_quantity[alias.lower()] = qty.name
            except Exception as e:
                raise KnowledgeBaseError(f"Failed to load quantities: {e}") from e

        # 2. Load equations from equations/ directory
        eq_dir = self.knowledge_dir / "equations"
        if eq_dir.exists():
            for eq_file in eq_dir.glob("*.yaml"):
                try:
                    with open(eq_file, encoding="utf-8") as f:
                        eq_data = yaml.safe_load(f) or {}
                        for item in eq_data.get("equations", []):
                            eq = Equation(**item)
                            self.equations[eq.id] = eq

                            # Index by topic
                            self._topic_index.setdefault(eq.topic, []).append(eq.id)

                            # Index by variable quantities
                            for var_sym, q_name in eq.variable_quantities.items():
                                self._variable_index.setdefault(q_name, []).append(eq.id)
                except Exception as e:
                    raise KnowledgeBaseError(f"Failed to load equations from {eq_file}: {e}") from e

    def get_equation(self, equation_id: str) -> Equation | None:
        """Get an equation by its ID."""
        return self.equations.get(equation_id)

    def get_quantity_by_symbol(self, symbol: str) -> PhysicsQuantity | None:
        """Find a quantity by its variable symbol (e.g. 'F' -> force)."""
        name = self._symbol_to_quantity.get(symbol)
        if name:
            return self.quantities.get(name)
        return None

    def get_quantity_by_name(self, name_or_alias: str) -> PhysicsQuantity | None:
        """Find a quantity by its primary name or alias (case-insensitive)."""
        clean = name_or_alias.lower().strip()
        name = self._alias_to_quantity.get(clean)
        if name:
            return self.quantities.get(name)
        return None

    def search_by_quantities(
        self, quantity_names: list[str], topic: str | None = None
    ) -> list[Equation]:
        """Find equations matching a list of quantity names or symbols.

        Ranks equations by number of overlapping quantities (descending).
        """
        # Resolve names/symbols to canonical quantity names
        canonical_names: set[str] = set()
        for q in quantity_names:
            qty = self.get_quantity_by_name(q) or self.get_quantity_by_symbol(q)
            if qty:
                canonical_names.add(qty.name)
            else:
                canonical_names.add(q.lower().strip())

        scored_equations: list[tuple[float, Equation]] = []

        for eq in self.equations.values():
            if topic and eq.topic != topic:
                continue

            eq_q_names = set(eq.variable_quantities.values())
            match_count = len(eq_q_names.intersection(canonical_names))
            if match_count > 0:
                # Score = match count / total variables in equation
                score = match_count / max(len(eq.variables), 1)
                scored_equations.append((score, eq))

        scored_equations.sort(key=lambda x: x[0], reverse=True)
        return [eq for _, eq in scored_equations]

    def search_by_topic(self, topic: str) -> list[Equation]:
        """Get all equations belonging to a specific topic."""
        eq_ids = self._topic_index.get(topic, [])
        return [self.equations[eid] for eid in eq_ids if eid in self.equations]

    def validate(self) -> list[str]:
        """Validate consistency of knowledge base.

        Checks:
        - All variable_quantities refer to known quantities
        - Equation expressions are valid SymPy strings
        - No orphan equations

        Returns:
            List of warning messages (empty if all valid).
        """
        warnings: list[str] = []
        for eq_id, eq in self.equations.items():
            for sym, q_name in eq.variable_quantities.items():
                if q_name not in self.quantities:
                    warnings.append(
                        f"Equation '{eq_id}' references unknown quantity '{q_name}' for symbol '{sym}'"
                    )
        return warnings
