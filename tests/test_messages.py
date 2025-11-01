from claireai.core.types.messages import (
    BaseMessage,
    SystemMessage,
    DeveloperMessage,
    UserMessage,
    AIMessage,
    ImagePart,
    TextPart,
)
from typing import List


def to_llm_msgs(messages: List[BaseMessage]) -> List[dict]:
    """Convert a list of BaseMessage instances to LLM-compatible dicts."""
    return [m.to_llm_dict() for m in messages]


def test_simple_conversation():
    sys_string = "You are a helpful assistant."
    user_string = "Hello, how can you assist me today?"
    dev_string = "This is a developer note."
    ai_string = "I can help you with various tasks."
    expected = [
        {"role": "system", "content": sys_string},
        {"role": "developer", "content": dev_string},
        {"role": "user", "content": user_string},
        {"role": "assistant", "content": ai_string},
    ]
    sys_msg = SystemMessage(content=sys_string)
    dev_msg = DeveloperMessage(content=dev_string)
    user_msg = UserMessage(content=user_string)
    ai_msg = AIMessage(content=ai_string)

    assert to_llm_msgs([sys_msg, dev_msg, user_msg, ai_msg]) == expected


def test_multipart_user_message_with_url():
    text_str = "Here is an image"
    image_url = "http://example.com/image.png"

    expected = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text_str},
                {"type": "input_image", "image_url": image_url, "detail": "high"},
            ],
        }
    ]

    user_msg = UserMessage(
        content=[
            TextPart(text=text_str),
            ImagePart(image_url=image_url, detail="high"),
        ]
    )

    assert to_llm_msgs([user_msg]) == expected
