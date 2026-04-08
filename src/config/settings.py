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
        "timeout_seconds": 600,
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
        "role_retries": {},
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
        "blocking_severities": ["critical", "major"],
        "enable_architect_quorum": True,
        "enable_system_runner": True,
        "enable_vector_memory": True,
        "embedding_model": "nomic-embed-text",
        "research_mode": False,
        "research_urls": [],
        "research_timeout_seconds": 10,
        "research_max_chars_per_source": 2000,
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

_REASONING_ROLES = (
    "ceo_planner",
    "market_researcher",
    "customer_support_feedback_analyst",
    "product_manager",
    "architect",
    "ui_ux_designer",
    "technical_writer",
    "release_manager",
)

_CODING_ROLES = (
    "database_engineer",
    "api_integration_engineer",
    "frontend_developer",
    "backend_developer",
    "data_analytics_engineer",
    "sre_reliability_engineer",
    "devops_engineer",
)

_CRITIC_ROLES = (
    "compliance_privacy_specialist",
    "performance_engineer",
    "security_engineer",
    "qa_engineer",
    "code_reviewer",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from *path* (defaults to ``config.yaml`` in project root).

    Unknown keys are silently ignored; missing keys fall back to *_DEFAULTS*.

    Environment variable overrides (all optional):

    ==========================================  ============================================
    Variable                                    Setting
    ==========================================  ============================================
    ``OLLAMA_BASE_URL`` / ``OLLAMA_URL``        ``llm.base_url``
    ``OLLAMA_MODEL``                            ``llm.model``
    ``MODEL_REASONING``                         overrides routing for reasoning/planning roles
    ``MODEL_CODING``                            overrides routing for implementation roles
    ``MODEL_CRITIC``                            overrides routing for reviewer/specialist roles
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
    _validate_config(merged)
    return merged


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate loaded config and raise ValueError with actionable messages."""
    llm_cfg = cfg.get("llm", {})
    if not isinstance(llm_cfg, dict):
        raise ValueError("Invalid config: 'llm' must be a mapping.")
    known_roles: set[str] = set()
    agents_cfg = cfg.get("agents", {})
    if isinstance(agents_cfg, dict):
        known_roles = {k for k in agents_cfg if isinstance(k, str) and k.strip()}

    allowed_raw = llm_cfg.get("allowed_models", [])
    if not isinstance(allowed_raw, list):
        raise ValueError("Invalid config: 'llm.allowed_models' must be a list.")
    allowed = {m.strip() for m in allowed_raw if isinstance(m, str) and m.strip()}

    def _validate_model_ref(model_name: object, where: str) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"Invalid config: '{where}' must be a non-empty model string.")
        if allowed and model_name.strip() not in allowed:
            raise ValueError(
                f"Invalid config: '{where}' uses '{model_name}', which is not in "
                f"llm.allowed_models={sorted(allowed)}"
            )

    _validate_model_ref(llm_cfg.get("model"), "llm.model")
    retries = llm_cfg.get("retries")
    if not isinstance(retries, int) or retries < 0:
        raise ValueError("Invalid config: 'llm.retries' must be a non-negative integer.")
    timeout_seconds = llm_cfg.get("timeout_seconds")
    if timeout_seconds is not None and (
        not isinstance(timeout_seconds, int) or timeout_seconds <= 0
    ):
        raise ValueError("Invalid config: 'llm.timeout_seconds' must be a positive integer.")

    routing = llm_cfg.get("routing", {})
    if not isinstance(routing, dict):
        raise ValueError("Invalid config: 'llm.routing' must be a mapping.")
    for role, model_name in routing.items():
        if not isinstance(role, str) or not role.strip():
            raise ValueError("Invalid config: all 'llm.routing' keys must be non-empty role names.")
        if known_roles and role not in known_roles:
            raise ValueError(
                f"Invalid config: 'llm.routing.{role}' references unknown role. "
                f"Known roles: {sorted(known_roles)}"
            )
        _validate_model_ref(model_name, f"llm.routing.{role}")

    fallbacks = llm_cfg.get("fallbacks", {})
    if not isinstance(fallbacks, dict):
        raise ValueError("Invalid config: 'llm.fallbacks' must be a mapping.")
    for role, models in fallbacks.items():
        if known_roles and role not in known_roles:
            raise ValueError(
                f"Invalid config: 'llm.fallbacks.{role}' references unknown role. "
                f"Known roles: {sorted(known_roles)}"
            )
        if not isinstance(models, list):
            raise ValueError(f"Invalid config: 'llm.fallbacks.{role}' must be a list.")
        for idx, model_name in enumerate(models):
            _validate_model_ref(model_name, f"llm.fallbacks.{role}[{idx}]")

    role_options = llm_cfg.get("role_options", {})
    if not isinstance(role_options, dict):
        raise ValueError("Invalid config: 'llm.role_options' must be a mapping.")
    for role, options in role_options.items():
        if known_roles and role not in known_roles:
            raise ValueError(
                f"Invalid config: 'llm.role_options.{role}' references unknown role. "
                f"Known roles: {sorted(known_roles)}"
            )
        if not isinstance(options, dict):
            raise ValueError(f"Invalid config: 'llm.role_options.{role}' must be a mapping.")
        num_predict = options.get("num_predict")
        if num_predict is not None and (not isinstance(num_predict, int) or num_predict <= 0):
            raise ValueError(
                f"Invalid config: 'llm.role_options.{role}.num_predict' must be a positive integer."
            )
        temperature = options.get("temperature")
        if temperature is not None and not isinstance(temperature, (int, float)):
            raise ValueError(
                f"Invalid config: 'llm.role_options.{role}.temperature' must be numeric."
            )

    role_retries = llm_cfg.get("role_retries", {})
    if not isinstance(role_retries, dict):
        raise ValueError("Invalid config: 'llm.role_retries' must be a mapping.")
    for role, retries in role_retries.items():
        if known_roles and role not in known_roles:
            raise ValueError(
                f"Invalid config: 'llm.role_retries.{role}' references unknown role. "
                f"Known roles: {sorted(known_roles)}"
            )
        if not isinstance(retries, int) or retries < 0:
            raise ValueError(
                f"Invalid config: 'llm.role_retries.{role}' must be a non-negative integer."
            )

    crew_cfg = cfg.get("crew", {})
    if not isinstance(crew_cfg, dict):
        raise ValueError("Invalid config: 'crew' must be a mapping.")
    max_fix_iterations = crew_cfg.get("max_fix_iterations")
    if not isinstance(max_fix_iterations, int) or max_fix_iterations < 0:
        raise ValueError(
            "Invalid config: 'crew.max_fix_iterations' must be a non-negative integer."
        )
    blocking = crew_cfg.get("blocking_severities", ["critical", "major"])
    if (
        not isinstance(blocking, list)
        or not blocking
        or not all(isinstance(item, str) and item.strip() for item in blocking)
    ):
        raise ValueError("Invalid config: 'crew.blocking_severities' must be a non-empty list.")
    research_timeout_seconds = crew_cfg.get("research_timeout_seconds")
    if not isinstance(research_timeout_seconds, int) or research_timeout_seconds <= 0:
        raise ValueError(
            "Invalid config: 'crew.research_timeout_seconds' must be a positive integer."
        )
    research_max_chars = crew_cfg.get("research_max_chars_per_source")
    if not isinstance(research_max_chars, int) or research_max_chars <= 0:
        raise ValueError(
            "Invalid config: 'crew.research_max_chars_per_source' must be a positive integer."
        )
    research_urls = crew_cfg.get("research_urls", [])
    if not isinstance(research_urls, list):
        raise ValueError("Invalid config: 'crew.research_urls' must be a list.")


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
    base_url = _str("OLLAMA_BASE_URL")
    if base_url is None:
        base_url = _str("OLLAMA_URL")
    if base_url is not None:
        cfg["llm"]["base_url"] = base_url
    model = _str("OLLAMA_MODEL")
    if model is not None:
        cfg["llm"]["model"] = model
    retries = _int("OLLAMA_RETRIES")
    if retries is not None:
        cfg["llm"]["retries"] = retries
    timeout = _int("OLLAMA_TIMEOUT")
    if timeout is not None:
        cfg["llm"]["timeout_seconds"] = timeout
    temperature = _float("OLLAMA_TEMPERATURE")
    if temperature is not None:
        cfg["llm"].setdefault("options", {})["temperature"] = temperature
    num_predict = _int("OLLAMA_NUM_PREDICT")
    if num_predict is not None:
        cfg["llm"].setdefault("options", {})["num_predict"] = num_predict

    # legacy role-group aliases
    model_reasoning = _str("MODEL_REASONING")
    model_coding = _str("MODEL_CODING")
    model_critic = _str("MODEL_CRITIC")
    if model is None:
        if model_reasoning:
            cfg["llm"]["model"] = model_reasoning
        routing = cfg["llm"].setdefault("routing", {})
        if model_reasoning:
            for role in _REASONING_ROLES:
                routing[role] = model_reasoning
        if model_coding:
            for role in _CODING_ROLES:
                routing[role] = model_coding
        if model_critic:
            for role in _CRITIC_ROLES:
                routing[role] = model_critic

    # crew overrides
    max_fix_iterations = _int("CREW_MAX_FIX_ITERATIONS")
    if max_fix_iterations is not None:
        cfg["crew"]["max_fix_iterations"] = max_fix_iterations
    stop_on_no_major = _bool("CREW_STOP_ON_NO_MAJOR_ISSUES")
    if stop_on_no_major is not None:
        cfg["crew"]["stop_on_no_major_issues"] = stop_on_no_major
    require_strategy_approval = _bool("CREW_REQUIRE_STRATEGY_APPROVAL")
    if require_strategy_approval is not None:
        cfg["crew"]["require_strategy_approval"] = require_strategy_approval

    # output overrides
    output_dir = _str("OUTPUT_DIR")
    if output_dir is not None:
        cfg["output"]["directory"] = output_dir
