"""Script to validate consistency and dimensional correctness of the physics knowledge base."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.solver.expression_parser import parse_equation_string
from physics_reasoning.units.dimension_checker import DimensionChecker
from physics_reasoning.units.unit_engine import UnitEngine


def validate_kb(kb_path: str = "data/knowledge") -> bool:
    """Validate all equations and quantities in the knowledge base.

    Returns:
        True if all valid, False if errors found.
    """
    print(f"Loading knowledge base from '{kb_path}'...")
    kb = KnowledgeBase(kb_path)
    try:
        kb.load()
    except Exception as e:
        print(f"FAILED to load knowledge base: {e}")
        return False

    print(f"Loaded {len(kb.quantities)} quantities and {len(kb.equations)} equations.")

    # 1. Structural consistency
    warnings = kb.validate()
    if warnings:
        print("\nKnowledge Base structural warnings:")
        for w in warnings:
            print(f"  [!] {w}")

    # 2. Equation parsing & Dimensional verification
    unit_engine = UnitEngine()
    dim_checker = DimensionChecker(unit_engine)

    has_errors = False
    print("\nVerifying equations...")

    for eq_id, eq in kb.equations.items():
        # Verify equation parses to SymPy
        try:
            parse_equation_string(eq.expression)
        except Exception as e:
            print(f"  [ERROR] Equation '{eq_id}' failed to parse: {e}")
            has_errors = True
            continue

        # Verify dimensional consistency using the quantities' units
        var_units: dict[str, str] = {}
        for var_sym, q_name in eq.variable_quantities.items():
            qty = kb.get_quantity_by_name(q_name)
            if qty and qty.si_unit:
                var_units[var_sym] = qty.si_unit

        dim_res = dim_checker.check_equation(eq.expression, var_units)
        if not dim_res.is_consistent:
            print(f"  [ERROR] Equation '{eq_id}' ({eq.expression}) dimensional mismatch: {dim_res.message}")
            has_errors = True
        else:
            print(f"  [OK] {eq_id:20s}: {eq.expression:35s} [{dim_res.lhs_dimension}]")

    if has_errors:
        print("\nValidation FAILED with errors.")
        return False

    print("\nKnowledge base validation PASSED successfully!")
    return True


if __name__ == "__main__":
    success = validate_kb()
    sys.exit(0 if success else 1)
