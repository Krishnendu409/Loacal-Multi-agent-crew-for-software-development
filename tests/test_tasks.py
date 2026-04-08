"""Unit tests for task definitions."""

from __future__ import annotations

import pytest

from src.tasks.software_dev_tasks import TASKS, Task


def test_all_expected_tasks_exist():
    expected_keys = {
        "ceo_planner",
        "market_researcher",
        "customer_support_feedback_analyst",
        "product_manager",
        "compliance_privacy_specialist",
        "architect",
        "ui_ux_designer",
        "database_engineer",
        "api_integration_engineer",
        "frontend_developer",
        "backend_developer",
        "data_analytics_engineer",
        "performance_engineer",
        "security_engineer",
        "qa_engineer",
        "code_reviewer",
        "technical_writer",
        "sre_reliability_engineer",
        "release_manager",
        "devops_engineer",
    }
    assert expected_keys == set(TASKS.keys())


def test_task_render_substitutes_requirements():
    task = TASKS["product_manager"]
    rendered = task.render(requirements="Build a todo app")
    assert "Build a todo app" in rendered
    assert "{requirements}" not in rendered


def test_task_render_missing_key_raises():
    task = TASKS["product_manager"]
    with pytest.raises(KeyError):
        task.render()  # missing 'requirements'


def test_task_has_non_empty_title_and_description():
    for key, task in TASKS.items():
        assert task.title, f"Task '{key}' has an empty title"
        assert task.description, f"Task '{key}' has an empty description"
