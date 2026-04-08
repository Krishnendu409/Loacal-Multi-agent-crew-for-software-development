"""Unit tests for the configuration loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.config.settings import load_config, _deep_merge


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested():
    base = {"llm": {"model": "mistral", "options": {"temperature": 0.7}}}
    override = {"llm": {"model": "llama3.2"}}
    result = _deep_merge(base, override)
    assert result["llm"]["model"] == "llama3.2"
    # options should be preserved from base
    assert result["llm"]["options"]["temperature"] == 0.7


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    _deep_merge(base, override)
    assert "y" not in base["a"]


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_returns_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["model"] == "qwen2.5:7b-instruct"
    assert cfg["llm"]["routing"]["customer_support_feedback_analyst"] == "qwen2.5:7b-instruct"
    assert cfg["llm"]["routing"]["compliance_privacy_specialist"] == "phi3:mini"
    assert cfg["llm"]["routing"]["ui_ux_designer"] == "qwen2.5:7b-instruct"
    assert cfg["llm"]["routing"]["database_engineer"] == "deepseek-coder:6.7b"
    assert cfg["llm"]["routing"]["backend_developer"] == "deepseek-coder:6.7b"
    assert cfg["llm"]["routing"]["performance_engineer"] == "phi3:mini"
    assert cfg["llm"]["routing"]["security_engineer"] == "phi3:mini"
    assert cfg["llm"]["fallbacks"]["backend_developer"] == [
        "qwen2.5:7b-instruct",
    ]
    assert cfg["llm"]["allowed_models"] == [
        "qwen2.5:7b-instruct",
        "deepseek-coder:6.7b",
        "phi3:mini",
    ]
    assert cfg["agents"]["ceo_planner"] is True
    assert cfg["agents"]["customer_support_feedback_analyst"] is True
    assert cfg["agents"]["ui_ux_designer"] is True
    assert cfg["agents"]["database_engineer"] is True
    assert cfg["agents"]["performance_engineer"] is True
    assert cfg["agents"]["security_engineer"] is True
    assert cfg["agents"]["technical_writer"] is True
    assert cfg["agents"]["sre_reliability_engineer"] is True
    assert cfg["agents"]["release_manager"] is True
    assert cfg["agents"]["product_manager"] is True
    assert cfg["crew"]["max_fix_iterations"] == 1
    assert cfg["crew"]["require_strategy_approval"] is True
    assert cfg["output"]["save_final_report"] is True
    assert cfg["skills"]["include_default_role_skills"] is True
    assert cfg["skills"]["enforce_handoff_sections"] is True
    assert cfg["skills"]["strict_mode"] is False
    assert cfg["skills"]["max_skills_per_agent"] == 12
    assert cfg["skills"]["packs"]["ecc"]["enabled"] is True
    assert cfg["skills"]["packs"]["ecc"]["profile"] == "starter"


def test_load_config_overrides_model(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """\
            llm:
              model: llama3.2
            """
        )
    )
    cfg = load_config(config_file)
    assert cfg["llm"]["model"] == "llama3.2"
    # defaults still present
    assert cfg["llm"]["base_url"] == "http://localhost:11434"


def test_load_config_disables_agent(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """\
            agents:
              product_manager: false
            """
        )
    )
    cfg = load_config(config_file)
    assert cfg["agents"]["product_manager"] is False
    assert cfg["agents"]["architect"] is True


def test_load_config_handles_empty_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    cfg = load_config(config_file)
    assert cfg["llm"]["model"] == "qwen2.5:7b-instruct"


def test_load_config_handles_invalid_yaml_type(tmp_path):
    """A YAML file that doesn't parse to a dict should fall back to defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n")
    cfg = load_config(config_file)
    assert cfg["llm"]["model"] == "qwen2.5:7b-instruct"


# ---------------------------------------------------------------------------
# P1 #8: environment variable overrides
# ---------------------------------------------------------------------------


def test_load_config_env_override_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "phi3:mini")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["model"] == "phi3:mini"


def test_load_config_env_override_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://myserver:11434")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["base_url"] == "http://myserver:11434"


def test_load_config_env_override_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_RETRIES", "3")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["retries"] == 3


def test_load_config_env_override_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT", "60")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["timeout_seconds"] == 60


def test_load_config_env_override_temperature(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.9")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["options"]["temperature"] == pytest.approx(0.9)


def test_load_config_env_override_num_predict(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "512")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["options"]["num_predict"] == 512


def test_load_config_env_override_max_fix_iterations(tmp_path, monkeypatch):
    monkeypatch.setenv("CREW_MAX_FIX_ITERATIONS", "5")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["crew"]["max_fix_iterations"] == 5


def test_load_config_env_override_stop_on_no_major_issues_false(tmp_path, monkeypatch):
    monkeypatch.setenv("CREW_STOP_ON_NO_MAJOR_ISSUES", "false")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["crew"]["stop_on_no_major_issues"] is False


def test_load_config_env_override_require_strategy_approval_false(tmp_path, monkeypatch):
    monkeypatch.setenv("CREW_REQUIRE_STRATEGY_APPROVAL", "0")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["crew"]["require_strategy_approval"] is False


def test_load_config_env_override_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/myoutput")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["output"]["directory"] == "/tmp/myoutput"


def test_load_config_env_overrides_take_precedence_over_file(tmp_path, monkeypatch):
    """Env vars should win over file-level settings."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("llm:\n  model: qwen2.5:7b-instruct\n")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-coder:6.7b")
    cfg = load_config(config_file)
    assert cfg["llm"]["model"] == "deepseek-coder:6.7b"
