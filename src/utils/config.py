
from pathlib import Path

import yaml


def load_config(config_path="configs/base.yaml"):
    """Load project settings from a YAML configuration file."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    return config
