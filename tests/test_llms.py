from claireai.core.configs import LLMConfigs
from claireai.core.llms import OpenAIChatLLM

from openai import AsyncOpenAI


class DummyAsyncOpenAI:
    """Dummy AsyncOpenAI client for testing."""

    pass


def test_openai_chat_llm_init_with_all_attributes():
    client = DummyAsyncOpenAI()
    model_name = "gpt-4"
    gen_args = {"temperature": 0.7, "max_tokens": 100}

    llm = OpenAIChatLLM(
        client=client,
        model_name=model_name,
        gen_args=gen_args,
    )

    assert llm._client is client
    assert llm.model_name == model_name
    assert llm.gen_args == gen_args


def test_openai_chat_llm_init_defaults_when_none():
    client = DummyAsyncOpenAI()
    model_name = "gpt-3.5-turbo"

    llm = OpenAIChatLLM(client=client, model_name=model_name)

    assert llm._client is client
    assert llm.model_name == model_name
    assert llm.gen_args == {}


def test_openai_chat_llm_from_config():
    client = AsyncOpenAI()
    model_name = "gpt-4"
    gen_args = {"temperature": 0.5}

    llm_config = LLMConfigs(
        client=client,
        model_name=model_name,
        gen_args=gen_args,
    )

    llm = OpenAIChatLLM.from_config(llm_config)

    assert isinstance(llm, OpenAIChatLLM)
    assert llm._client is client
    assert llm.model_name == model_name
    assert llm.gen_args == gen_args
