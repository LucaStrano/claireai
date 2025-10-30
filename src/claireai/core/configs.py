from pydantic import BaseModel, Field
from typing import Any


class AgentConfigs(BaseModel):
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
