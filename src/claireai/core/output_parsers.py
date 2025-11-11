from typing import Generic, TypeVar, Optional
import abc

from claireai.core.types.responses import (
    LLMResponse,
    LLMCompletionUsage,
    LLMToolCall,
    Structure,
    FinishReason,
)

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

ResponseT = TypeVar("ResponseT")


class OutputParser(abc.ABC, Generic[ResponseT]):
    """
    Base class for parsing LLM responses into the unified LLMResponse format.
    """

    @abc.abstractmethod
    def parse(self, response: ResponseT) -> LLMResponse:
        """
        Parse a raw LLM response into an LLMResponse object.

        Args:
            response (ResponseT): The raw response from the LLM API.

        Returns:
            LLMResponse: The parsed response.
        """
        pass

    @abc.abstractmethod
    def parse_structured(
        self, response: ResponseT, structure: type[Structure]
    ) -> LLMResponse[Structure]:
        """
        Parse a structured output response into an LLMResponse object.

        Args:
            response: The raw response from the LLM API.
            structure: The expected structure type.

        Returns:
            LLMResponse[Structure]: The parsed response with structured content.
        """
        pass


class OpenAIOutputParser(OutputParser[ChatCompletion]):
    """Output parser for OpenAI API responses."""

    def _parse_usage(
        self, response: ChatCompletion | ParsedChatCompletion
    ) -> Optional[LLMCompletionUsage]:
        """Extract usage statistics from response."""
        if hasattr(response, "usage"):
            return LLMCompletionUsage(
                completion_tokens=getattr(response.usage, "completion_tokens", None),
                prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                total_tokens=getattr(response.usage, "total_tokens", None),
            )
        return None

    def _parse_tool_calls(self, message: ChatCompletionMessage) -> list[LLMToolCall]:
        """Extract tool calls from message."""
        return [
            LLMToolCall(
                id=tool.id,
                name=tool.function.name,
                arguments=tool.function.arguments,
            )
            for tool in getattr(message, "tool_calls", []) or []
        ]

    def parse(self, response: ChatCompletion) -> LLMResponse:
        """
        Parse an OpenAI ChatCompletion response.

        Args:
            response: The ChatCompletion response from OpenAI API.

        Returns:
            LLMResponse: The parsed response.
        """
        c = response.choices[0]
        return LLMResponse(
            completion_id=response.id,
            content=getattr(c.message, "content", "") or "",
            model=response.model,
            finish_reason=c.finish_reason or FinishReason.MISSING,
            reasoning=getattr(c.message, "reasoning", None),
            parsed_content=None,
            tool_calls=self._parse_tool_calls(c.message),
            completion_usage=self._parse_usage(response),
        )

    def parse_structured(
        self, response: ParsedChatCompletion, structure: type[Structure]
    ) -> LLMResponse[Structure]:
        """
        Parse an OpenAI ParsedChatCompletion response with structured output.

        Args:
            response (ParsedChatCompletion): The ParsedChatCompletion response from OpenAI API.
            structure (type[Structure]): The expected structure type.

        Returns:
            LLMResponse[Structure]: The parsed response with structured content.
        """
        # TODO: parsed should be extracted from an output type tool call
        c = response.choices[0]
        return LLMResponse[structure](
            completion_id=response.id,
            content=getattr(c.message, "content", "") or "",
            model=response.model,
            finish_reason=c.finish_reason or FinishReason.MISSING,
            reasoning=getattr(c.message, "reasoning", None),
            parsed_content=c.message.parsed,
            tool_calls=self._parse_tool_calls(c.message),
            completion_usage=self._parse_usage(response),
        )
