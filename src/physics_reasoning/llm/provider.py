"""LLM provider abstraction and implementations."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from physics_reasoning.core.exceptions import (
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from physics_reasoning.core.models import LLMResponse


class LLMProvider(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send completion request to LLM."""
        ...

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Whether this provider/model natively supports function/tool calling."""
        ...

    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether this provider/model natively supports JSON response_format."""
        ...


class LiteLLMProvider(LLMProvider):
    """Unified LLM provider using litellm (supports Ollama, OpenAI, Anthropic, HuggingFace, etc.)."""

    def __init__(
        self,
        model_name: str = "ollama/phi3:mini",
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
    )
    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Call LiteLLM with structured output or tool definitions."""
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if response_format:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                kwargs["response_format"] = response_format
            elif isinstance(response_format, dict):
                kwargs["response_format"] = response_format

        start_time = time.perf_counter()
        try:
            raw_res = litellm.completion(**kwargs)
        except litellm.RateLimitError as e:
            raise LLMRateLimitError(f"LLM rate limit reached: {e}") from e
        except litellm.Timeout as e:
            raise LLMTimeoutError(f"LLM call timed out after {self.timeout}s: {e}") from e
        except Exception as e:
            raise LLMError(f"LiteLLM error: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        choices = raw_res.choices if hasattr(raw_res, "choices") else []
        if not choices:
            raise LLMEmptyResponseError("LLM returned no choices")

        choice = choices[0]
        message = choice.message if hasattr(choice, "message") else None
        content = message.content or "" if message else ""

        # Extract tool calls if any
        tool_calls: list[dict[str, Any]] = []
        if message and hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(raw_res, "usage") and raw_res.usage:
            usage = {
                "prompt_tokens": getattr(raw_res.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(raw_res.usage, "completion_tokens", 0),
                "total_tokens": getattr(raw_res.usage, "total_tokens", 0),
            }

        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=self.model_name,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )

    def supports_tool_calling(self) -> bool:
        import litellm

        try:
            return litellm.supports_function_calling(model=self.model_name)
        except Exception:
            return False

    def supports_structured_output(self) -> bool:
        import litellm

        try:
            return litellm.supports_response_schema(model=self.model_name)
        except Exception:
            return False


class MockLLMProvider(LLMProvider):
    """Deterministic Mock LLM provider for fast testing without external API calls."""

    def __init__(self, responses: list[str | dict[str, Any]] | None = None):
        self.responses: list[str | dict[str, Any]] = responses or []
        self.call_index: int = 0
        self.call_history: list[dict[str, Any]] = []

    def set_responses(self, responses: list[str | dict[str, Any]]) -> None:
        self.responses = responses
        self.call_index = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "response_format": response_format,
            }
        )

        if not self.responses:
            raise LLMEmptyResponseError("MockLLMProvider has no responses configured")

        resp = self.responses[min(self.call_index, len(self.responses) - 1)]
        self.call_index += 1

        if isinstance(resp, str):
            content = resp
            tool_calls: list[dict[str, Any]] = []
        elif isinstance(resp, dict):
            content = resp.get("content", "")
            tool_calls = resp.get("tool_calls", [])
        else:
            content = str(resp)
            tool_calls = []

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock/test-model",
            finish_reason="stop",
            latency_ms=10.0,
        )

    def supports_tool_calling(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return True
