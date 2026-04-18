"""Configuration loading/saving helpers.

These helpers keep YAML interactions centralized for CLI and scripts.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path where YAML should be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Saved config to {output_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        New merged configuration where `override_config` values win.
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
