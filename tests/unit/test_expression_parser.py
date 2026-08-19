"""Tests for expression parser."""

from __future__ import annotations

import pytest
from sympy import Eq, Symbol

from physics_reasoning.core.exceptions import ExpressionParseError
from physics_reasoning.solver.expression_parser import (
    extract_symbol_names,
    extract_symbols,
    parse_equation_string,
    parse_expression,
)


class TestExpressionParser:
    def test_parse_simple_expression(self):
        expr = parse_expression("m * a")
        assert extract_symbol_names(expr) == ["a", "m"]

    def test_parse_implicit_multiplication(self):
        expr = parse_expression("2 a")
        assert extract_symbol_names(expr) == ["a"]

    def test_parse_subscripted_symbols(self):
        expr = parse_expression("v_f - v_i")
        assert extract_symbol_names(expr) == ["v_f", "v_i"]

    def test_parse_power_syntax(self):
        expr1 = parse_expression("v^2")
        expr2 = parse_expression("v**2")
        assert expr1 == expr2

    def test_parse_equation_string(self):
        eq = parse_equation_string("F = m * a")
        assert isinstance(eq, Eq)
        assert extract_symbol_names(eq) == ["F", "a", "m"]

    def test_parse_equation_complex(self):
        eq = parse_equation_string("d = v_i * t + (1/2) * a * t^2")
        assert "v_i" in extract_symbol_names(eq)
        assert "t" in extract_symbol_names(eq)
        assert "a" in extract_symbol_names(eq)
        assert "d" in extract_symbol_names(eq)

    def test_empty_string_raises(self):
        with pytest.raises(ExpressionParseError):
            parse_expression("")

    def test_missing_equals_raises(self):
        with pytest.raises(ExpressionParseError):
            parse_equation_string("F m a")

    def test_multiple_equals_raises(self):
        with pytest.raises(ExpressionParseError):
            parse_equation_string("a = b = c")

    def test_disallowed_patterns_rejected(self):
        with pytest.raises(ExpressionParseError):
            parse_expression("__import__('os').system('ls')")

        with pytest.raises(ExpressionParseError):
            parse_expression("eval('1+1')")

        with pytest.raises(ExpressionParseError):
            parse_expression("open('secret.txt')")
