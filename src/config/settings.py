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
        "model": "qwen2.5:7b-instruct",
        "base_url": "http://localhost:11434",
        "retries": 1,
        "timeout_seconds": 120,
        "allowed_models": [
            "qwen2.5:7b-instruct",
            "deepseek-coder:6.7b",
            "phi3:mini",
        ],
        "options": {
            "temperature": 0.4,
            "num_predict": 2048,
        },
        "routing": {
            "ceo_planner": "qwen2.5:7b-instruct",
            "market_researcher": "qwen2.5:7b-instruct",
            "product_manager": "qwen2.5:7b-instruct",
            "architect": "qwen2.5:7b-instruct",
            "frontend_developer": "deepseek-coder:6.7b",
            "backend_developer": "deepseek-coder:6.7b",
            "qa_engineer": "phi3:mini",
            "code_reviewer": "phi3:mini",
            "devops_engineer": "deepseek-coder:6.7b",
        },
        "fallbacks": {
            "ceo_planner": ["phi3:mini"],
            "market_researcher": ["phi3:mini"],
            "product_manager": ["phi3:mini"],
            "architect": ["phi3:mini"],
            "frontend_developer": ["qwen2.5:7b-instruct"],
            "backend_developer": ["qwen2.5:7b-instruct"],
            "qa_engineer": ["qwen2.5:7b-instruct"],
            "code_reviewer": ["qwen2.5:7b-instruct"],
            "devops_engineer": ["qwen2.5:7b-instruct"],
        },
        "role_options": {
            "ceo_planner": {"num_predict": 1536, "temperature": 0.3},
            "market_researcher": {"num_predict": 1536, "temperature": 0.2},
            "product_manager": {"num_predict": 1536, "temperature": 0.3},
            "architect": {"num_predict": 1536, "temperature": 0.2},
            "frontend_developer": {"num_predict": 2048, "temperature": 0.2},
            "backend_developer": {"num_predict": 2048, "temperature": 0.2},
            "qa_engineer": {"num_predict": 1024, "temperature": 0.1},
            "code_reviewer": {"num_predict": 1024, "temperature": 0.1},
            "devops_engineer": {"num_predict": 1024, "temperature": 0.2},
        },
    },
    "agents": {
        "ceo_planner": True,
        "market_researcher": True,
        "product_manager": True,
        "architect": True,
        "frontend_developer": True,
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
        "require_strategy_approval": True,
    },
    "skills": {
        "include_default_role_skills": True,
        "enforce_handoff_sections": True,
        "strict_mode": False,
        "max_skills_per_agent": 12,
        "packs": {
            "ecc": {
                "enabled": True,
                "profile": "starter",
            }
        },
        "shared": [
            "structured communication",
            "security-first thinking",
            "verification mindset",
            "documentation discipline",
        ],
        "per_role": {},
        "include": [],
        "per_role_include": {},
        "exclude": [],
        "per_role_exclude": {},
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
