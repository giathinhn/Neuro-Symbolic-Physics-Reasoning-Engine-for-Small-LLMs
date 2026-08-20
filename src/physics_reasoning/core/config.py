"""Configuration loading and validation for the physics reasoning engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from physics_reasoning.core.exceptions import ConfigError


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    # LLM settings
    model_name: str = "ollama_chat/qwen2.5:3b"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: float = 60.0

    # Pipeline settings
    max_retries: int = 3
    max_tool_calls_per_attempt: int = 10

    # Paths
    knowledge_base_path: str = "data/knowledge"
    dataset_path: str = "data/problems"

    # Verification toggles
    enable_dimensional_check: bool = True
    enable_unit_check: bool = True
    enable_arithmetic_check: bool = True
    enable_bounds_check: bool = True
    enable_substitution_check: bool = True
    enable_consistency_check: bool = True
    enable_equation_validity_check: bool = True

    # Numerical settings
    numerical_tolerance: float = 1e-6
    answer_tolerance: float = 0.01
    solver_timeout_seconds: float = 10.0

    # Reproducibility
    random_seed: int = 42

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    # LLM mode
    use_tool_calling: bool = False  # False = structured output mode


def load_config(config_path: str | None = None) -> PipelineConfig:
    """Load configuration from YAML file, environment variables, and defaults.

    Priority: environment variables > config file > defaults.

    Args:
        config_path: Path to YAML configuration file. If None, uses defaults.

    Returns:
        Validated PipelineConfig.

    Raises:
        ConfigError: If config file exists but is invalid.
    """
    config_data: dict[str, Any] = {}

    # Load from YAML if path provided
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        try:
            with open(path) as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    config_data.update(loaded)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse config file {config_path}: {e}") from e

    # Override from environment variables
    env_mapping: dict[str, str] = {
        "DEFAULT_MODEL": "model_name",
        "TEMPERATURE": "temperature",
        "MAX_RETRIES": "max_retries",
        "TIMEOUT_SECONDS": "timeout_seconds",
        "LOG_LEVEL": "log_level",
        "LOG_FILE": "log_file",
        "RANDOM_SEED": "random_seed",
    }

    for env_key, config_key in env_mapping.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            # Convert to appropriate type
            field_info = PipelineConfig.model_fields.get(config_key)
            if field_info is not None:
                annotation = field_info.annotation
                if annotation is float:
                    config_data[config_key] = float(env_val)
                elif annotation is int:
                    config_data[config_key] = int(env_val)
                elif annotation is bool:
                    config_data[config_key] = env_val.lower() in ("true", "1", "yes")
                else:
                    config_data[config_key] = env_val

    try:
        return PipelineConfig(**config_data)
    except Exception as e:
        raise ConfigError(f"Invalid configuration: {e}") from e
