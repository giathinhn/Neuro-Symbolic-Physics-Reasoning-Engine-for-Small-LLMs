"""Tool definitions and executor for LLM function calling mode."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from physics_reasoning.core.exceptions import ToolExecutionError
from physics_reasoning.core.models import ToolCallRecord
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.solver.expression_parser import parse_expression
from physics_reasoning.solver.numerical import evaluate_numeric
from physics_reasoning.solver.symbolic_solver import SymbolicSolver
from physics_reasoning.units.dimension_checker import DimensionChecker
from physics_reasoning.units.unit_engine import UnitEngine

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_equations",
            "description": "Search the physics knowledge base for equations matching given physical quantities or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Physical quantity names or symbols (e.g. ['force', 'mass', 'acceleration'])",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional physics topic (e.g. 'kinematics', 'newton_laws', 'work_energy')",
                    },
                },
                "required": ["quantities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "Solve a single equation or system of equations symbolically and evaluate with known values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Equations as strings (e.g. ['F = m * a'])",
                    },
                    "known_values": {
                        "type": "object",
                        "description": "Mapping of known variable symbols to numeric values (e.g. {'F': 10.0, 'm': 2.0})",
                    },
                    "target_variable": {
                        "type": "string",
                        "description": "Variable symbol to solve for (e.g. 'a')",
                    },
                },
                "required": ["equations", "known_values", "target_variable"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert a numerical value from one unit to another (e.g. km/h to m/s).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value"},
                    "from_unit": {"type": "string", "description": "Source unit (e.g. 'km/h')"},
                    "to_unit": {"type": "string", "description": "Target unit (e.g. 'm/s')"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_dimensions",
            "description": "Check if an equation is dimensionally consistent given units for each variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "Equation string (e.g. 'F = m * a')"},
                    "variable_units": {
                        "type": "object",
                        "description": "Mapping of variable symbols to units (e.g. {'F': 'N', 'm': 'kg', 'a': 'm/s**2'})",
                    },
                },
                "required": ["equation", "variable_units"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_solution",
            "description": "Verify that variable values satisfy an equation upon back-substitution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "Equation string"},
                    "variable_values": {
                        "type": "object",
                        "description": "Mapping of all variable symbols to values",
                    },
                },
                "required": ["equation", "variable_values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression numerically (e.g. '10 / 2' -> 5.0).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression string"},
                },
                "required": ["expression"],
            },
        },
    },
]


class ToolExecutor:
    """Execute LLM tool calls against the underlying symbolic engines."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        solver: SymbolicSolver | None = None,
        unit_engine: UnitEngine | None = None,
        dimension_checker: DimensionChecker | None = None,
    ):
        self.kb = knowledge_base or KnowledgeBase()
        self.solver = solver or SymbolicSolver()
        self.unit_engine = unit_engine or UnitEngine()
        self.dimension_checker = dimension_checker or DimensionChecker(self.unit_engine)

        self._tool_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "search_equations": self._handle_search_equations,
            "solve_equation": self._handle_solve_equation,
            "convert_units": self._handle_convert_units,
            "check_dimensions": self._handle_check_dimensions,
            "verify_solution": self._handle_verify_solution,
            "calculate": self._handle_calculate,
        }

    def execute(self, tool_name: str, arguments: dict[str, Any] | str) -> ToolCallRecord:
        """Execute a tool by name with arguments dict."""
        start_time = time.perf_counter()

        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments)
            except Exception:
                args_dict = {}
        else:
            args_dict = arguments

        handler = self._tool_handlers.get(tool_name)
        if not handler:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=args_dict,
                result=None,
                error=f"Unknown tool: '{tool_name}'",
                duration_ms=duration_ms,
                success=False,
            )

        try:
            result = handler(args_dict)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=args_dict,
                result=result,
                error=None,
                duration_ms=duration_ms,
                success=True,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=args_dict,
                result=None,
                error=str(e),
                duration_ms=duration_ms,
                success=False,
            )

    def _handle_search_equations(self, args: dict[str, Any]) -> dict[str, Any]:
        quantities = args.get("quantities", [])
        topic = args.get("topic")
        matches = self.kb.search_by_quantities(quantities, topic=topic)
        return {
            "equations": [
                {
                    "id": eq.id,
                    "name": eq.name,
                    "expression": eq.expression,
                    "variables": eq.variables,
                    "topic": eq.topic,
                }
                for eq in matches[:5]
            ]
        }

    def _handle_solve_equation(self, args: dict[str, Any]) -> dict[str, Any]:
        eqs = args.get("equations", [])
        known_values = args.get("known_values", {})
        target = args.get("target_variable", "")
        # Cast values to float
        numeric_knowns = {k: float(v) for k, v in known_values.items()}
        res = self.solver.solve_system(eqs, numeric_knowns, target)
        return {
            "target_variable": res.target_variable,
            "solutions": res.solutions,
            "is_numeric": res.is_numeric,
            "warnings": res.warnings,
        }

    def _handle_convert_units(self, args: dict[str, Any]) -> dict[str, Any]:
        val = float(args.get("value", 0.0))
        from_u = str(args.get("from_unit", ""))
        to_u = str(args.get("to_unit", ""))
        res = self.unit_engine.convert(val, from_u, to_u)
        return {
            "value": res.to_value,
            "unit": to_u,
            "from_value": res.from_value,
            "from_unit": from_u,
        }

    def _handle_check_dimensions(self, args: dict[str, Any]) -> dict[str, Any]:
        eq = str(args.get("equation", ""))
        var_units = args.get("variable_units", {})
        res = self.dimension_checker.check_equation(eq, var_units)
        return {
            "is_consistent": res.is_consistent,
            "lhs_dimension": res.lhs_dimension,
            "rhs_dimension": res.rhs_dimension,
            "message": res.message,
        }

    def _handle_verify_solution(self, args: dict[str, Any]) -> dict[str, Any]:
        eq = str(args.get("equation", ""))
        var_values = {k: float(v) for k, v in args.get("variable_values", {}).items()}
        is_sat, diff = self.solver.verify_substitution(eq, var_values)
        return {
            "is_satisfied": is_sat,
            "residual": diff,
        }

    def _handle_calculate(self, args: dict[str, Any]) -> dict[str, Any]:
        expr_str = str(args.get("expression", ""))
        parsed = parse_expression(expr_str)
        val = evaluate_numeric(parsed)
        return {"result": val}
