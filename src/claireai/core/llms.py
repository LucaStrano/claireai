from typing import List, Optional, Any, Dict
import abc

from claireai.core.types.messages import BaseMessage

from pydantic import BaseModel

from openai import AsyncOpenAI


class LLM(abc.ABC):
    """
    Base Class for Language Models.

    The concern of this class is to adapt the injected client's API
    into claireAI's unified response interface.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        gen_args: Dict[str, Any] = {},
    ):
        self._client = client
        self.model_name = model_name
        self.gen_args = gen_args

    @abc.abstractmethod
    async def generate(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[Any]] = None,  # TODO: Define a BaseTool class
    ) -> Any:
        """
        Generate a response.

        Args:
            messages (List[BaseMessage]): The list of prompt messages.
            tools (List[Any]): A list of tools.
        Returns:
            Any: Response.
        """
        pass

    @abc.abstractmethod
    async def generate_with_structured_output(
        self,
        messages: list[BaseMessage],
        structure: BaseModel,
        tools: Optional[List[Any]] = None,
    ) -> Any:
        """
        Generate a structured response.

        Args:
            messages (List[BaseMessage]): The list of prompt messages.
            structure (BaseModel): The Pydantic model defining the output structure.
            tools (List[Any]): A list of tools.
        Returns:
            Any: Response.
        """
        pass


class OpenAIChatLLM(LLM):
    """OpenAI Chat Model."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        gen_args: Dict[str, Any] = {},
    ):
        super().__init__(client, model_name, gen_args)

    async def generate(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[Any]] = None,  # TODO: Define a BaseTool class
    ) -> Any:
        """
        Generate a response.

        Args:
            messages (List[BaseMessage]): The list of prompt messages.
            tools (List[Any]): A list of tools.
        Returns:
            Any: Response.
        """
        response = await self._client.chat.completions.create(
            messages=[m.to_llm_dict() for m in messages],
            model=self.model_name,
            **self.gen_args,
        )
        return response  # TODO create a proper response object

    async def generate_with_structured_output(
        self,
        messages: list[BaseMessage],
        structure: BaseModel,
        tools: Optional[List[Any]] = None,
    ) -> Any:
        """
        Generate a structured response.

        Args:
            messages (List[BaseMessage]): The list of prompt messages.
            structure (BaseModel): The Pydantic model defining the output structure.
            tools (List[Any]): A list of tools.
        Returns:
            Any: Response.
        """
        response = await self._client.chat.completions.parse(
            model=self.model_name,
            messages=[m.to_llm_dict() for m in messages],
            response_format=structure,
            **self.gen_args,
        )

        return response  # TODO create a proper response object
