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
            "customer_support_feedback_analyst": "qwen2.5:7b-instruct",
            "product_manager": "qwen2.5:7b-instruct",
            "compliance_privacy_specialist": "phi3:mini",
            "architect": "qwen2.5:7b-instruct",
            "ui_ux_designer": "qwen2.5:7b-instruct",
            "database_engineer": "deepseek-coder:6.7b",
            "api_integration_engineer": "deepseek-coder:6.7b",
            "frontend_developer": "deepseek-coder:6.7b",
            "backend_developer": "deepseek-coder:6.7b",
            "data_analytics_engineer": "deepseek-coder:6.7b",
            "performance_engineer": "phi3:mini",
            "security_engineer": "phi3:mini",
            "qa_engineer": "phi3:mini",
            "code_reviewer": "phi3:mini",
            "technical_writer": "qwen2.5:7b-instruct",
            "sre_reliability_engineer": "deepseek-coder:6.7b",
            "release_manager": "qwen2.5:7b-instruct",
            "devops_engineer": "deepseek-coder:6.7b",
        },
        "fallbacks": {
            "ceo_planner": ["phi3:mini"],
            "market_researcher": ["phi3:mini"],
            "customer_support_feedback_analyst": ["phi3:mini"],
            "product_manager": ["phi3:mini"],
            "compliance_privacy_specialist": ["qwen2.5:7b-instruct"],
            "architect": ["phi3:mini"],
            "ui_ux_designer": ["phi3:mini"],
            "database_engineer": ["qwen2.5:7b-instruct"],
            "api_integration_engineer": ["qwen2.5:7b-instruct"],
            "frontend_developer": ["qwen2.5:7b-instruct"],
            "backend_developer": ["qwen2.5:7b-instruct"],
            "data_analytics_engineer": ["qwen2.5:7b-instruct"],
            "performance_engineer": ["qwen2.5:7b-instruct"],
            "security_engineer": ["qwen2.5:7b-instruct"],
            "qa_engineer": ["qwen2.5:7b-instruct"],
            "code_reviewer": ["qwen2.5:7b-instruct"],
            "technical_writer": ["phi3:mini"],
            "sre_reliability_engineer": ["qwen2.5:7b-instruct"],
            "release_manager": ["phi3:mini"],
            "devops_engineer": ["qwen2.5:7b-instruct"],
        },
        "role_options": {
            "ceo_planner": {"num_predict": 1536, "temperature": 0.3},
            "market_researcher": {"num_predict": 1536, "temperature": 0.2},
            "customer_support_feedback_analyst": {"num_predict": 1280, "temperature": 0.2},
            "product_manager": {"num_predict": 1536, "temperature": 0.3},
            "compliance_privacy_specialist": {"num_predict": 1024, "temperature": 0.1},
            "architect": {"num_predict": 1536, "temperature": 0.2},
            "ui_ux_designer": {"num_predict": 1536, "temperature": 0.3},
            "database_engineer": {"num_predict": 1536, "temperature": 0.2},
            "api_integration_engineer": {"num_predict": 1536, "temperature": 0.2},
            "frontend_developer": {"num_predict": 2048, "temperature": 0.2},
            "backend_developer": {"num_predict": 2048, "temperature": 0.2},
            "data_analytics_engineer": {"num_predict": 1536, "temperature": 0.2},
            "performance_engineer": {"num_predict": 1024, "temperature": 0.1},
            "security_engineer": {"num_predict": 1024, "temperature": 0.1},
            "qa_engineer": {"num_predict": 1024, "temperature": 0.1},
            "code_reviewer": {"num_predict": 1024, "temperature": 0.1},
            "technical_writer": {"num_predict": 1536, "temperature": 0.2},
            "sre_reliability_engineer": {"num_predict": 1024, "temperature": 0.2},
            "release_manager": {"num_predict": 1024, "temperature": 0.2},
            "devops_engineer": {"num_predict": 1024, "temperature": 0.2},
        },
    },
    "agents": {
        "ceo_planner": True,
        "market_researcher": True,
        "customer_support_feedback_analyst": True,
        "product_manager": True,
        "compliance_privacy_specialist": True,
        "architect": True,
        "ui_ux_designer": True,
        "database_engineer": True,
        "api_integration_engineer": True,
        "frontend_developer": True,
        "backend_developer": True,
        "data_analytics_engineer": True,
        "performance_engineer": True,
        "security_engineer": True,
        "qa_engineer": True,
        "code_reviewer": True,
        "technical_writer": True,
        "sre_reliability_engineer": True,
        "release_manager": True,
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

    Environment variable overrides (all optional):

    ==========================================  ============================================
    Variable                                    Setting
    ==========================================  ============================================
    ``OLLAMA_BASE_URL``                         ``llm.base_url``
    ``OLLAMA_MODEL``                            ``llm.model``
    ``OLLAMA_RETRIES``                          ``llm.retries`` (integer)
    ``OLLAMA_TIMEOUT``                          ``llm.timeout_seconds`` (integer)
    ``OLLAMA_TEMPERATURE``                      ``llm.options.temperature`` (float)
    ``OLLAMA_NUM_PREDICT``                      ``llm.options.num_predict`` (integer)
    ``CREW_MAX_FIX_ITERATIONS``                 ``crew.max_fix_iterations`` (integer)
    ``CREW_STOP_ON_NO_MAJOR_ISSUES``            ``crew.stop_on_no_major_issues`` (bool)
    ``CREW_REQUIRE_STRATEGY_APPROVAL``          ``crew.require_strategy_approval`` (bool)
    ``OUTPUT_DIR``                              ``output.directory``
    ==========================================  ============================================
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    file_config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                file_config = loaded

    merged = _deep_merge(_DEFAULTS, file_config)
    _apply_env_overrides(merged)
    return merged


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    """Mutate *cfg* in-place to apply environment variable overrides."""

    def _str(var: str) -> str | None:
        return os.environ.get(var) or None

    def _int(var: str) -> int | None:
        val = os.environ.get(var)
        if val is None:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    def _float(var: str) -> float | None:
        val = os.environ.get(var)
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def _bool(var: str) -> bool | None:
        val = os.environ.get(var)
        if val is None:
            return None
        stripped = val.strip().lower()
        if not stripped:
            return None
        return stripped not in ("0", "false", "no", "off")

    # llm overrides
    if (v := _str("OLLAMA_BASE_URL")) is not None:
        cfg["llm"]["base_url"] = v
    if (v := _str("OLLAMA_MODEL")) is not None:
        cfg["llm"]["model"] = v
    if (v := _int("OLLAMA_RETRIES")) is not None:
        cfg["llm"]["retries"] = v
    if (v := _int("OLLAMA_TIMEOUT")) is not None:
        cfg["llm"]["timeout_seconds"] = v
    if (v := _float("OLLAMA_TEMPERATURE")) is not None:
        cfg["llm"].setdefault("options", {})["temperature"] = v
    if (v := _int("OLLAMA_NUM_PREDICT")) is not None:
        cfg["llm"].setdefault("options", {})["num_predict"] = v

    # crew overrides
    if (v := _int("CREW_MAX_FIX_ITERATIONS")) is not None:
        cfg["crew"]["max_fix_iterations"] = v
    if (v := _bool("CREW_STOP_ON_NO_MAJOR_ISSUES")) is not None:
        cfg["crew"]["stop_on_no_major_issues"] = v
    if (v := _bool("CREW_REQUIRE_STRATEGY_APPROVAL")) is not None:
        cfg["crew"]["require_strategy_approval"] = v

    # output overrides
    if (v := _str("OUTPUT_DIR")) is not None:
        cfg["output"]["directory"] = v
