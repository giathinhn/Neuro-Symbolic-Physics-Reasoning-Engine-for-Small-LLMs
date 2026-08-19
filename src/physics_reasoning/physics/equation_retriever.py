"""Equation retrieval and semantic matching against knowledge base."""

from __future__ import annotations

from sympy import Eq, simplify

from physics_reasoning.core.models import Equation
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.solver.expression_parser import parse_equation_string


class EquationRetriever:
    """Retrieve and match equations from the knowledge base."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def retrieve_by_quantities(
        self,
        quantity_names: list[str],
        topic: str | None = None,
        top_k: int = 5,
    ) -> list[Equation]:
        """Retrieve top-k equations matching a set of quantities.

        Args:
            quantity_names: List of quantity names or symbols.
            topic: Optional topic filter.
            top_k: Max equations to return.

        Returns:
            List of matching Equation objects.
        """
        results = self.kb.search_by_quantities(quantity_names, topic=topic)
        return results[:top_k]

    def match_expression(self, expression_str: str) -> Equation | None:
        """Match a free-form equation string to a knowledge base equation.

        Strategy:
        1. Try exact string match.
        2. Try SymPy structural algebraic equivalence:
           Eq(A, B) <=> A - B == 0
           Check if simplify((lhs1 - rhs1) - (lhs2 - rhs2)) == 0 or
           simplify((lhs1 - rhs1) + (lhs2 - rhs2)) == 0
        """
        if not expression_str or "=" not in expression_str:
            return None

        # 1. Exact string match
        clean = expression_str.replace(" ", "")
        for eq in self.kb.equations.values():
            if eq.expression.replace(" ", "") == clean:
                return eq

        # 2. SymPy algebraic equivalence match
        try:
            parsed_input = parse_equation_string(expression_str)
            input_diff = parsed_input.lhs - parsed_input.rhs

            for eq in self.kb.equations.values():
                try:
                    kb_parsed = parse_equation_string(eq.expression)
                    kb_diff = kb_parsed.lhs - kb_parsed.rhs

                    if simplify(input_diff - kb_diff) == 0 or simplify(input_diff + kb_diff) == 0:
                        return eq
                except Exception:
                    continue
        except Exception:
            return None

        return None

    def suggest_equations(
        self, given_quantities: list[str], target_quantity: str
    ) -> list[Equation]:
        """Find equations that connect the given quantities to the target quantity.

        Uses set matching: finds equations that contain the target AND at least one given quantity.
        """
        target_qty = self.kb.get_quantity_by_name(target_quantity) or self.kb.get_quantity_by_symbol(target_quantity)
        target_name = target_qty.name if target_qty else target_quantity

        given_canonical: set[str] = set()
        for g in given_quantities:
            qty = self.kb.get_quantity_by_name(g) or self.kb.get_quantity_by_symbol(g)
            given_canonical.add(qty.name if qty else g)

        matching_eqs: list[Equation] = []
        for eq in self.kb.equations.values():
            var_names = set(eq.variable_quantities.values())
            if target_name in var_names and len(var_names.intersection(given_canonical)) > 0:
                matching_eqs.append(eq)

        return matching_eqs
