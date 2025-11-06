from pydantic import BaseModel, Field
from typing import Any, Literal, Optional, TypeVar, Generic
from dataclasses import dataclass
from enum import StrEnum


Structure = TypeVar("Structure", bound=BaseModel)


@dataclass
class LLMCompletionUsage:
    """Token usage statistics for the completion."""

    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class LLMToolCall:
    """Tool call made by the LLM during generation."""

    id: str
    name: str
    type: Literal["function"] = "function"
    arguments: Optional[dict[str, Any]] = None


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"
    MISSING = "missing"


@dataclass
class LLMResponse(Generic[Structure]):
    """Standardized response from an LLM generation."""

    completion_id: str
    content: str
    model: str
    finish_reason: FinishReason = FinishReason.MISSING
    reasoning: Optional[str] = None
    parsed_content: Optional[Structure] = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    completion_usage: Optional[LLMCompletionUsage] = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if there are any tool calls in the response.

        Returns:
            bool: True if there are tool calls, False otherwise.
        """
        return len(self.tool_calls) > 0
