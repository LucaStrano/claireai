from pathlib import Path

from claireai.core.agent import Agent
from claireai.core.configs import AgentConfigs


class AgentFactory:
    """Factory class for creating Agent instances."""

    @staticmethod
    def create(**kwargs) -> Agent:
        """Create an Agent instance with the provided parameters.
        Args:
            **kwargs: Parameters for the Agent instance.
        Returns:
            Agent: A configured Agent instance.
        """
        return Agent(**kwargs)

    @staticmethod
    def from_config(config: AgentConfigs) -> Agent:
        """Create an Agent instance from an AgentConfig instance.
        Args:
            config (AgentConfigs): The config used for the agent.
        Returns:
            Agent: A configured Agent instance.
        """
        return Agent.from_config(config)

    @staticmethod
    def from_config_file(
        config_path: str | Path,
    ) -> Agent:
        """Create an Agent instance from a config.yml file path.
        Args:
            config_path (str | Path): The YAML config path used for the agent.
        """
        return Agent.from_config_file(config_path)

    @staticmethod
    def create_batch(
        configs: list[AgentConfigs],
    ) -> list[Agent]:
        """Create multiple Agent instances from a list of configs.
        Args:
            configs (list[AgentConfigs]): List of agent configurations.
        """
        return [AgentFactory.from_config(config) for config in configs]

    @staticmethod
    def create_batch_from_files(
        config_paths: list[str | Path],
    ) -> list[Agent]:
        """Create multiple Agent instances from a list of config file paths.
        Args:
            config_paths (list[str | Path]): List of YAML config file paths.
        """
        return [AgentFactory.from_config_file(path) for path in config_paths]
