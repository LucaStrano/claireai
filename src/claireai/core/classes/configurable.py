from pydantic import BaseModel
from pydantic_core import PydanticSerializationError


class Configurable:
    """A Base Class for Objects that can be created from Configs."""

    @classmethod
    def from_config(cls, config: BaseModel) -> "Configurable":
        """Create an instance of `Configurable` from a Config Model.
        Args:
            config (BaseModel): The config used to create the instance.
        Raises:
            ValueError: If the config is invalid.
        Returns:
            Configurable: An instance of `Configurable`.
        """
        try:
            return cls(**config.model_dump(mode="python", warnings="error"))
        except (PydanticSerializationError, TypeError) as e:
            raise ValueError(
                f"Error creating {cls.__name__} from Configs, "
                "make sure the Config object is valid.",
            ) from e
