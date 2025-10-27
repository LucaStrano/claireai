from yaml import safe_load, YAMLError
from json import dumps

from claireai.core.configs import AgentConfigs


def yaml_to_agent_configs(yaml_path: str) -> AgentConfigs:
    try:
        with open(yaml_path, "r") as file:
            yaml_content = file.read()
        yaml_dict = safe_load(yaml_content)
        print(dumps(yaml_dict, indent=2))

    except YAMLError as e:
        import os
        import pathlib

        parent_dir_path = pathlib.Path(__file__).parent.resolve()
        parent_dir_name = parent_dir_path.name
        print(parent_dir_name)
        print(parent_dir_path)
        agent_name = None
        # if the parent dir contains a .py file with the same name, assume we are in an agent directory
        if f"{parent_dir_name}.py" in os.listdir(parent_dir_path):
            agent_name = parent_dir_name
        raise YAMLError(
            "Error parsing config.yml"
            + (f" for agent '{agent_name}', " if agent_name else ", ")
            + "Please ensure the YAML syntax is correct."
        ) from e

    try:
        agent_conf = yaml_dict["agent_configs"]
    except KeyError as e:
        raise KeyError(
            "The YAML file is missing the 'agent_configs' key. "
            "Please ensure the configuration file is structured correctly."
        ) from e

    # TODO create llm instance from configs
    llm = object()

    # TODO create strategy instance from configs
    strategy = object()

    # TODO create tools instances from configs
    tools = []

    # TODO create delegation instance from configs
    delegations = []

    # TODO create handoffs instances from configs
    handoffs = []

    # TODO create prompts instances from configs
    prompts = {}

    try:
        return AgentConfigs(
            name=agent_conf["name"],
            description=agent_conf["description"],
            is_callable=agent_conf["is_callable"],
            llm=llm,
            strategy=strategy,
            tools=tools,
            delegations=delegations,
            handoffs=handoffs,
            prompts=prompts,
            extras=agent_conf.get("extras", {}),
        )
    except KeyError as e:
        raise KeyError(
            f"The 'agent_configs' section is missing the key: {e}. "
            "Please ensure all required fields are present."
        ) from e
