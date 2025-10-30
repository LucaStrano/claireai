from pydantic import Field
from pydantic_core import PydanticSerializationError

from typing import Optional, Any

from .configs import AgentConfigs
from claireai.utils.converters import yaml_to_agent_configs

from pathlib import Path

AGENT_DEFAULT_DESCRIPTION: str = "No description provided."


class Agent:
    """Base class for all agents."""

    name: str
    """The name of the agent."""

    description: Optional[str] = None
    """The description of the agent. This is used by other agents that call this one 
    to understand its purpose, both for handoff and delegation."""

    llm: Any  # Placeholder for LLM instance
    """The language model used by the agent."""

    strategy: Any  # Placeholder for strategy instance
    """The strategy used by the agent to determine its actions."""

    is_callable: bool = False
    """Whether to generate an API endpoint for this agent when running `claire run`
    at project level."""

    tools: list[Any] = Field(default_factory=list)  # Placeholder for tool instances
    """The tools available to the agent."""

    handoffs: list[Any] = Field(
        default_factory=list
    )  # Placeholder for handoff instances
    """The Agents this agent can pass control to."""

    prompts: dict[str, str] = Field(
        default_factory=dict
    )  # Placeholder for prompt templates
    """The prompt templates used by the agent."""

    extras: dict[str, Any] = Field(default_factory=dict)
    """A dictionary to hold Any extra configurations or parameters."""

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        is_callable: bool = False,
        llm: Any = None,
        strategy: Any = None,
        tools: Optional[list[Any]] = None,
        handoffs: Optional[list[Any]] = None,
        prompts: Optional[dict[str, str]] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = (
            description if description is not None else AGENT_DEFAULT_DESCRIPTION
        )
        self.is_callable = is_callable
        self.llm = llm
        self.strategy = strategy
        self.tools = tools if tools is not None else []
        self.handoffs = handoffs if handoffs is not None else []
        self.prompts = prompts if prompts is not None else {}
        self.extras = extras if extras is not None else {}

    @classmethod
    def from_config(cls, config: AgentConfigs) -> "Agent":
        """Create an Agent instance from an AgentConfig instance.
        Args:
            config (AgentConfigs): The config used for the agent.
        Raises:
            ValueError: If the config is invalid.
        """
        try:
            agent = cls(**config.model_dump(warnings="error"))
            return agent
        except PydanticSerializationError as e:
            raise ValueError(
                f"Error creating Agent{f' {config.name}' if config.name else ''} from config file,"
                "make sure the config is valid.",
            ) from e

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "Agent":
        """Create an Agent instance from a config.yml file path.
        Args:
            config_path (str | Path): The YAML config used for the agent.
        Raises:
            YAMLError: If the config file is malformed.
            KeyError: If required keys are missing in the config file.

        """
        config = yaml_to_agent_configs(config_path)
        return cls.from_config(config)
