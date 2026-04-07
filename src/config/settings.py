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
        "retries": 1,
        "timeout_seconds": 120,
        "options": {
            "temperature": 0.4,
            "num_predict": 2048,
        },
        "routing": {
            "product_manager": "qwen2.5:7b-instruct",
            "architect": "qwen2.5:7b-instruct",
            "backend_developer": "qwen2.5-coder:7b",
            "qa_engineer": "phi3:mini",
            "code_reviewer": "phi3:mini",
            "devops_engineer": "llama3.2:3b",
        },
        "fallbacks": {
            "product_manager": ["llama3.2:3b"],
            "architect": ["llama3.2:3b"],
            "backend_developer": ["deepseek-coder:6.7b", "llama3.2:3b"],
            "qa_engineer": ["llama3.2:3b"],
            "code_reviewer": ["llama3.2:3b"],
            "devops_engineer": ["phi3:mini"],
        },
        "role_options": {
            "product_manager": {"num_predict": 1536, "temperature": 0.3},
            "architect": {"num_predict": 1536, "temperature": 0.2},
            "backend_developer": {"num_predict": 2048, "temperature": 0.2},
            "qa_engineer": {"num_predict": 1024, "temperature": 0.1},
            "code_reviewer": {"num_predict": 1024, "temperature": 0.1},
            "devops_engineer": {"num_predict": 1024, "temperature": 0.2},
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
    "crew": {
        "max_fix_iterations": 1,
        "stop_on_no_major_issues": True,
    },
    "skills": {
        "include_default_role_skills": True,
        "enforce_handoff_sections": True,
        "shared": [
            "structured communication",
            "security-first thinking",
            "verification mindset",
            "documentation discipline",
        ],
        "per_role": {},
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
