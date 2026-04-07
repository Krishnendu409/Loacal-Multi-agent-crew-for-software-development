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
    assert cfg["llm"]["routing"]["ui_ux_designer"] == "qwen2.5:7b-instruct"
    assert cfg["llm"]["routing"]["backend_developer"] == "deepseek-coder:6.7b"
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
    assert cfg["agents"]["ui_ux_designer"] is True
    assert cfg["agents"]["security_engineer"] is True
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
