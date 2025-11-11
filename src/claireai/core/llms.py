from typing import List, Optional, Any, Dict, TypeVar, Generic
import abc

from claireai.core.classes.configurable import Configurable
from claireai.core.types.messages import BaseMessage
from claireai.core.output_parsers import OutputParser, OpenAIOutputParser
from claireai.core.types.responses import (
    LLMResponse,
    # LLMCompletionUsage,
    # LLMToolCall,
    Structure,
    # FinishReason,
)

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

ClientT = TypeVar("ClientT")


class LLM(abc.ABC, Generic[ClientT], Configurable):
    """
    Base Class for Language Models.

    The concern of this class is to adapt the injected client's API
    into claireAI's unified response interface by using an OutputParser.
    """

    def __init__(
        self,
        client: ClientT,
        model_name: str,
        output_parser: OutputParser,
        gen_args: Dict[str, Any] = {},
    ):
        self._client: ClientT = client
        self.model_name = model_name
        self._output_parser = output_parser
        self.gen_args = gen_args

    @abc.abstractmethod
    async def generate(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[Any]] = None,  # TODO: Define a BaseTool class
    ) -> LLMResponse:
        """
        Generate a response.

        Args:
            messages (List[BaseMessage]): The list of prompt messages.
            tools (List[Any]): A list of tools.
        Returns:
            LLMResponse: Response.
        """
        pass

    @abc.abstractmethod
    async def generate_with_structured_output(
        self,
        messages: list[BaseMessage],
        structure: type[Structure],
        tools: Optional[List[Any]] = None,
    ) -> LLMResponse[Structure]:
        """
        Generate a structured response.

        Args:
            messages (List[BaseMessage]): The list of prompt messages.
            structure (BaseModel): The Pydantic model defining the output structure.
            tools (List[Any]): A list of tools.
        Returns:
            LLMResponse[Structure]: Response.
        """
        pass


class OpenAIChatLLM(LLM[AsyncOpenAI]):
    """OpenAI Chat Model."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        output_parser: Optional[OpenAIOutputParser] = None,
        gen_args: Dict[str, Any] = {},
    ):
        super().__init__(
            client, model_name, output_parser or OpenAIOutputParser(), gen_args
        )

    async def generate(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[Any]] = None,  # TODO: Define a BaseTool class
    ) -> LLMResponse:
        """
        Generate a response.

        Args:
            messages (List[BaseMessage]): The list of prompt messages.
            tools (List[Any]): A list of tools.
        Returns:
            LLMResponse: Parsed response from the API.
        """
        res: ChatCompletion = await self._client.chat.completions.create(
            messages=[m.to_llm_dict() for m in messages],
            model=self.model_name,
            **self.gen_args,
        )
        return self._output_parser.parse(res)

    async def generate_with_structured_output(
        self,
        messages: list[BaseMessage],
        structure: type[Structure],
        tools: Optional[List[Any]] = None,
    ) -> LLMResponse[Structure]:
        """
        Generate a structured response.

        Args:
            messages (List[BaseMessage]): The list of prompt messages.
            structure (BaseModel): The Pydantic model defining the output structure.
            tools (List[Any]): A list of tools.
        Returns:
            LLMResponse[Structure]: Parsed response from the API.
        """
        res: ParsedChatCompletion = await self._client.chat.completions.parse(
            model=self.model_name,
            messages=[m.to_llm_dict() for m in messages],
            response_format=structure,
            **self.gen_args,
        )
        # TODO: for now, structured output works without tools.
        # With tools implemented, structured output will be handled as a "finish_tool".
        return self._output_parser.parse_structured(res, structure)
