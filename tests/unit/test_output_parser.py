"""Tests for LLM output parser."""

from __future__ import annotations

import pytest

from physics_reasoning.core.exceptions import LLMOutputParseError
from physics_reasoning.llm.output_parser import parse_llm_output, validate_parsed_output


class TestOutputParser:
    def test_parse_clean_json(self):
        json_text = """{
            "problem_understanding": "A force on mass problem",
            "quantities": [
                {"name": "mass", "symbol": "m", "value": 2.0, "unit": "kg", "role": "given"},
                {"name": "force", "symbol": "F", "value": 10.0, "unit": "N", "role": "given"},
                {"name": "acceleration", "symbol": "a", "role": "target"}
            ],
            "equations": [
                {"equation_id": "newton2", "expression": "F = m * a"}
            ],
            "target_variable": "a"
        }"""
        parsed = parse_llm_output(json_text)
        assert len(parsed.quantities) == 3
        assert parsed.target_variable == "a"

    def test_parse_markdown_code_block(self):
        md_text = """Here is the extracted analysis:
```json
{
    "problem_understanding": "Kinematics velocity calculation",
    "quantities": [
        {"name": "displacement", "symbol": "d", "value": 100.0, "unit": "m", "role": "given"},
        {"name": "time", "symbol": "t", "value": 5.0, "unit": "s", "role": "given"},
        {"name": "velocity", "symbol": "v", "role": "target"}
    ],
    "equations": [
        {"expression": "v = d / t"}
    ],
    "target_variable": "v"
}
```
Hope this helps!"""
        parsed = parse_llm_output(md_text)
        assert parsed.target_variable == "v"
        assert len(parsed.equations) == 1

    def test_parse_surrounded_by_text(self):
        surrounded = """Sure, here is the answer:
{
    "quantities": [{"name": "mass", "symbol": "m", "value": 5.0}],
    "equations": [{"expression": "F = m * a"}]
}
Let me know if you need more details."""
        parsed = parse_llm_output(surrounded)
        assert len(parsed.quantities) == 1

    def test_parse_garbage_raises(self):
        with pytest.raises(LLMOutputParseError):
            parse_llm_output("This is pure natural language with no JSON structure whatsoever.")
