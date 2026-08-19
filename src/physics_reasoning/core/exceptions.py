"""Custom exception hierarchy for the physics reasoning engine."""

from __future__ import annotations


class PhysicsEngineError(Exception):
    """Base exception for all physics engine errors."""

    pass


class ConfigError(PhysicsEngineError):
    """Error in configuration loading or validation."""

    pass


class LLMError(PhysicsEngineError):
    """Base error for LLM-related issues."""

    pass


class LLMTimeoutError(LLMError):
    """LLM call timed out."""

    pass


class LLMEmptyResponseError(LLMError):
    """LLM returned an empty response."""

    pass


class LLMOutputParseError(LLMError):
    """Failed to parse LLM output into structured form."""

    def __init__(self, message: str, raw_output: str | None = None):
        super().__init__(message)
        self.raw_output = raw_output


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    pass


class SolverError(PhysicsEngineError):
    """Base error for symbolic solver issues."""

    pass


class ExpressionParseError(SolverError):
    """Failed to parse a mathematical expression."""

    def __init__(self, message: str, expression: str | None = None):
        super().__init__(message)
        self.expression = expression


class SolverTimeoutError(SolverError):
    """Solver exceeded time limit."""

    pass


class NoSolutionError(SolverError):
    """No solution found for the equation system."""

    pass


class UnitError(PhysicsEngineError):
    """Base error for unit-related issues."""

    pass


class UnitParseError(UnitError):
    """Failed to parse a unit string."""

    def __init__(self, message: str, unit_string: str | None = None):
        super().__init__(message)
        self.unit_string = unit_string


class UnitConversionError(UnitError):
    """Failed to convert between units (incompatible dimensions)."""

    def __init__(self, message: str, from_unit: str | None = None, to_unit: str | None = None):
        super().__init__(message)
        self.from_unit = from_unit
        self.to_unit = to_unit


class VerificationFailedError(PhysicsEngineError):
    """Verification pipeline detected errors."""

    pass


class DatasetError(PhysicsEngineError):
    """Error in dataset loading or validation."""

    pass


class KnowledgeBaseError(PhysicsEngineError):
    """Error in knowledge base loading or querying."""

    pass


class ToolExecutionError(PhysicsEngineError):
    """Error executing a tool call."""

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message)
        self.tool_name = tool_name


class PipelineError(PhysicsEngineError):
    """Error in the pipeline orchestrator."""

    pass
