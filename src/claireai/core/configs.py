from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Generic

from claireai.core.llms import LLM, ClientT


class AgentConfigs(BaseModel):
    """Configuration for an Agent instance."""

    name: str
    description: str = "No description provided."
    is_callable: bool = False
    llm: Any = None  # Placeholder for LLM instance
    strategy: Any = None  # Placeholder for strategy instance
    tools: list[Any] = Field(default_factory=list)  # Placeholder for tool instances
    handoffs: list[Any] = Field(
        default_factory=list
    )  # Placeholder for handoff instances
    prompts: dict[str, str] = Field(
        default_factory=dict
    )  # Placeholder for prompt templates
    extras: dict[str, Any] = Field(default_factory=dict)


class LLMConfigs(BaseModel, Generic[ClientT]):
    """Configuration for an LLM instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: LLM[ClientT]
    model_name: str
    gen_args: dict[str, Any] = Field(default_factory=dict)
