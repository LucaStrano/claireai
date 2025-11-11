import pytest
from unittest.mock import Mock

from claireai.core.types.responses import (
    LLMResponse,
    LLMCompletionUsage,
    LLMToolCall,
    FinishReason,
)
from claireai.core.output_parsers import OpenAIOutputParser


@pytest.fixture
def parser():
    return OpenAIOutputParser()


def create_mock_usage(completion_tokens=10, prompt_tokens=20, total_tokens=30):
    """Helper to create mock usage object."""
    usage = Mock()
    usage.completion_tokens = completion_tokens
    usage.prompt_tokens = prompt_tokens
    usage.total_tokens = total_tokens
    return usage


def create_mock_message(content="test content", reasoning=None, tool_calls=None):
    """Helper to create mock message object."""
    message = Mock()
    message.content = content
    message.reasoning = reasoning
    message.tool_calls = tool_calls or []
    return message


def create_mock_choice(message=None, finish_reason="stop"):
    """Helper to create mock choice object."""
    choice = Mock()
    choice.message = message or create_mock_message()
    choice.finish_reason = finish_reason
    return choice


def create_mock_chat_completion(
    id="completion-123",
    model="gpt-4",
    choices=None,
    usage=None,
):
    """Helper to create mock ChatCompletion object."""
    response = Mock()
    response.id = id
    response.model = model
    response.choices = choices or [create_mock_choice()]
    response.usage = usage or create_mock_usage()
    return response


class TestParseUsage:
    def test_parse_usage_with_valid_data(self, parser):
        """Test parsing usage statistics from response."""
        response = create_mock_chat_completion(usage=create_mock_usage(5, 15, 20))
        result = parser._parse_usage(response)

        assert isinstance(result, LLMCompletionUsage)
        assert result.completion_tokens == 5
        assert result.prompt_tokens == 15
        assert result.total_tokens == 20

    def test_parse_usage_missing_attribute(self, parser):
        """Test parsing when usage attribute is missing."""
        response = Mock(spec=[])  # No usage attribute
        result = parser._parse_usage(response)

        assert result is None

    def test_parse_usage_with_none_values(self, parser):
        """Test parsing usage with None values."""
        usage = Mock()
        usage.completion_tokens = None
        usage.prompt_tokens = None
        usage.total_tokens = None
        response = create_mock_chat_completion(usage=usage)
        result = parser._parse_usage(response)

        assert result.completion_tokens is None
        assert result.prompt_tokens is None
        assert result.total_tokens is None


class TestParseToolCalls:
    def test_parse_tool_calls_empty(self, parser):
        """Test parsing when no tool calls present."""
        message = create_mock_message(tool_calls=None)
        result = parser._parse_tool_calls(message)

        assert result == []

    def test_parse_tool_calls_with_single_tool(self, parser):
        """Test parsing with a single tool call."""
        tool = Mock()
        tool.id = "call-123"
        tool.function.name = "get_weather"
        tool.function.arguments = '{"location": "NYC", "celsius": 22}'

        message = create_mock_message(tool_calls=[tool])
        result = parser._parse_tool_calls(message)

        assert len(result) == 1
        assert isinstance(result[0], LLMToolCall)
        assert result[0].id == "call-123"
        assert result[0].name == "get_weather"
        assert result[0].arguments == '{"location": "NYC", "celsius": 22}'

    def test_parse_tool_calls_with_multiple_tools(self, parser):
        """Test parsing with multiple tool calls."""
        tool1 = Mock()
        tool1.id = "call-1"
        tool1.function.name = "get_weather"
        tool1.function.arguments = '{"location": "NYC"}'

        tool2 = Mock()
        tool2.id = "call-2"
        tool2.function.name = "get_time"
        tool2.function.arguments = '{"timezone": "EST"}'

        message = create_mock_message(tool_calls=[tool1, tool2])
        result = parser._parse_tool_calls(message)

        assert len(result) == 2
        assert result[0].name == "get_weather"
        assert result[1].name == "get_time"


class TestParse:
    def test_parse_basic_response(self, parser):
        """Test parsing a basic ChatCompletion response."""
        message = create_mock_message(content="Hello, how can I help?")
        choice = create_mock_choice(message=message, finish_reason="stop")
        response = create_mock_chat_completion(choices=[choice])

        result = parser.parse(response)

        assert isinstance(result, LLMResponse)
        assert result.completion_id == "completion-123"
        assert result.content == "Hello, how can I help?"
        assert result.model == "gpt-4"
        assert result.finish_reason == "stop"
        assert result.reasoning is None
        assert result.parsed_content is None
        assert result.tool_calls == []

    def test_parse_response_with_reasoning(self, parser):
        """Test parsing response with reasoning."""
        message = create_mock_message(
            content="Answer", reasoning="Let me think through this..."
        )
        choice = create_mock_choice(message=message)
        response = create_mock_chat_completion(choices=[choice])

        result = parser.parse(response)

        assert result.reasoning == "Let me think through this..."
        assert result.content == "Answer"

    def test_parse_response_with_tool_calls(self, parser):
        """Test parsing response with tool calls."""
        tool = Mock()
        tool.id = "call-123"
        tool.function.name = "search"
        tool.function.arguments = '{"query": "AI"}'

        message = create_mock_message(tool_calls=[tool])
        choice = create_mock_choice(message=message)
        response = create_mock_chat_completion(choices=[choice])

        result = parser.parse(response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_parse_response_with_missing_content(self, parser):
        """Test parsing when content attribute is missing."""
        message = Mock(spec=[])  # No content
        choice = create_mock_choice(message=message)
        response = create_mock_chat_completion(choices=[choice])

        result = parser.parse(response)

        assert result.content == ""

    def test_parse_response_finish_reason_default(self, parser):
        """Test parsing with missing finish_reason defaults."""
        choice = Mock()
        choice.message = create_mock_message()
        choice.finish_reason = None  # Missing
        response = create_mock_chat_completion(choices=[choice])

        result = parser.parse(response)

        assert result.finish_reason == FinishReason.MISSING


class TestParseStructured:
    def test_parse_structured_basic(self, parser):
        """Test parsing structured response."""
        parsed_data = Mock()
        message = create_mock_message(content="Structured output")
        message.parsed = parsed_data

        choice = create_mock_choice(message=message)
        response = create_mock_chat_completion(choices=[choice])

        result = parser.parse_structured(response, type(parsed_data))

        assert isinstance(result, LLMResponse)
        assert result.content == "Structured output"
        assert result.parsed_content is parsed_data
        assert result.tool_calls == []

    def test_parse_structured_with_all_fields(self, parser):
        """Test parsing structured response with all fields populated."""
        parsed_data = {"key": "value"}
        message = create_mock_message(content="Result", reasoning="Analyzed the input")
        message.parsed = parsed_data

        choice = create_mock_choice(message=message, finish_reason="stop")
        usage = create_mock_usage(100, 50, 150)
        response = create_mock_chat_completion(
            id="comp-456", model="gpt-4-turbo", choices=[choice], usage=usage
        )

        result = parser.parse_structured(response, dict)

        assert result.completion_id == "comp-456"
        assert result.model == "gpt-4-turbo"
        assert result.reasoning == "Analyzed the input"
        assert result.parsed_content == parsed_data
        assert result.completion_usage.total_tokens == 150
