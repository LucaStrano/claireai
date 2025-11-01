from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Literal, Annotated, Optional, Union, List
from dataclasses import dataclass


class _PartType(StrEnum):
    """Enumeration of possible types of parts in a multi-part message."""

    INPUT_TEXT = "input_text"
    INPUT_IMAGE = "input_image"


# === Content Parts ===
class TextPart(BaseModel):
    """Text part of a multi-part message."""

    type: Literal[_PartType.INPUT_TEXT] = _PartType.INPUT_TEXT
    text: str


class ImagePart(BaseModel):
    """Image part of a multi-part message."""

    type: Literal[_PartType.INPUT_IMAGE] = _PartType.INPUT_IMAGE
    image_url: str  # TODO Add base64 support
    detail: Optional[Literal["low", "high", "auto"]] = None


# TODO Add other parts: file, audio...

# ContentPart Union used for Typing
_ContentPart = Annotated[Union[TextPart, ImagePart], Field(discriminator="type")]


class MessageRole(StrEnum):
    """Enumeration of possible message roles."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class BaseMessage(BaseModel):
    """Base Class for messages."""

    role: MessageRole
    content: Union[str, List[_ContentPart]]

    def to_llm_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


@dataclass
class SystemMessage:
    """System message."""

    content: str

    def to_llm_dict(self):
        return BaseMessage(role=MessageRole.SYSTEM, content=self.content).to_llm_dict()


@dataclass
class DeveloperMessage:
    """Developer message."""

    content: str

    def to_llm_dict(self):
        return BaseMessage(
            role=MessageRole.DEVELOPER, content=self.content
        ).to_llm_dict()


@dataclass
class UserMessage:
    """User message."""

    content: Union[str, List[_ContentPart]]

    def to_llm_dict(self):
        return BaseMessage(role=MessageRole.USER, content=self.content).to_llm_dict()


@dataclass
class AIMessage:
    """AI message."""

    content: str  # TODO Add support for multi-part assistant messages

    def to_llm_dict(self):
        return BaseMessage(
            role=MessageRole.ASSISTANT, content=self.content
        ).to_llm_dict()


@dataclass
class ToolMessage:
    """Tool message."""

    content: str  # TODO Add support for tool attributes

    def to_llm_dict(self):
        return BaseMessage(role=MessageRole.TOOL, content=self.content).to_llm_dict()
