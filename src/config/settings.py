"""Configuration loader – reads config.yaml and exposes typed settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Default path: project root / config.yaml
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


_DEFAULTS: dict[str, Any] = {
    "llm": {
        "model": "mistral",
        "base_url": "http://localhost:11434",
        "options": {
            "temperature": 0.7,
            "num_predict": 2048,
        },
    },
    "agents": {
        "product_manager": True,
        "architect": True,
        "backend_developer": True,
        "qa_engineer": True,
        "code_reviewer": True,
        "devops_engineer": False,
    },
    "output": {
        "directory": "output",
        "save_individual_responses": True,
        "save_final_report": True,
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from *path* (defaults to ``config.yaml`` in project root).

    Unknown keys are silently ignored; missing keys fall back to *_DEFAULTS*.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    file_config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                file_config = loaded

    return _deep_merge(_DEFAULTS, file_config)
