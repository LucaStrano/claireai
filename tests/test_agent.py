from claireai.core.agent import Agent, AGENT_DEFAULT_DESCRIPTION
from claireai.core.configs import AgentConfigs

from pathlib import Path


class DummyLLM:
    pass


class DummyStrategy:
    pass


def test_agent_init_with_all_attributes():
    llm = DummyLLM()
    strategy = DummyStrategy()
    tools = [object(), object()]
    handoffs = ["handoff_a"]
    prompts = {"system": "You are an helpful assistant."}
    extras = {"extra1": 1, "extra2": "extra_value"}

    agent = Agent(
        name="TestAgent",
        description="A test agent",
        is_callable=True,
        llm=llm,
        strategy=strategy,
        tools=tools,
        handoffs=handoffs,
        prompts=prompts,
        extras=extras,
    )

    assert agent.name == "TestAgent"
    assert agent.description == "A test agent"
    assert agent.is_callable is True
    assert agent.llm is llm
    assert agent.strategy is strategy
    assert agent.tools == tools
    assert agent.handoffs == handoffs
    assert agent.prompts == prompts
    assert agent.extras == extras


def test_agent_init_defaults_when_none():
    llm = DummyLLM()
    strategy = DummyStrategy()
    agent = Agent(name="Alpha", llm=llm, strategy=strategy)

    assert agent.name == "Alpha"
    # description=None should fall back to the default string
    assert agent.description == AGENT_DEFAULT_DESCRIPTION
    assert agent.is_callable is False
    assert agent.llm is llm
    assert agent.strategy is strategy
    assert agent.tools == []
    assert agent.handoffs == []
    assert agent.prompts == {}
    assert agent.extras == {}


def test_agent_from_config_full():

    llm = DummyLLM()
    strategy = DummyStrategy()

    cfg = AgentConfigs(
        name="ConfigAgent",
        description="From config",
        is_callable=True,
        llm=llm,
        strategy=strategy,
        tools=["t1", "t2"],
        handoffs=["h1", "h2"],
        prompts={"system": "Be concise."},
        extras={"extra": 1},
    )

    agent = Agent.from_config(cfg)

    assert isinstance(agent, Agent)
    assert agent.name == cfg.name
    assert agent.description == cfg.description
    assert agent.is_callable is True
    assert agent.llm is llm
    assert agent.strategy is strategy
    assert agent.tools == ["t1", "t2"]
    assert agent.handoffs == ["h1", "h2"]
    assert agent.prompts == {"system": "Be concise."}
    assert agent.extras == {"extra": 1}


def test_agent_from_yaml_configs():
    from claireai.utils.converters import yaml_to_agent_configs

    yaml_path = Path(__file__).parent / "fixtures" / "example_agent.yml"

    config_from_path = yaml_to_agent_configs(yaml_path)
    config_from_str = yaml_to_agent_configs(str(yaml_path))

    agent_from_conf_instance = Agent.from_config(config_from_path)
    agent_from_conf_file = Agent.from_config_file(yaml_path)

    assert isinstance(config_from_path, AgentConfigs)
    assert isinstance(config_from_str, AgentConfigs)
    assert isinstance(agent_from_conf_instance, Agent)
    assert isinstance(agent_from_conf_file, Agent)
