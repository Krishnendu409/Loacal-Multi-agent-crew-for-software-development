"""Unit tests for the DevCrew orchestrator.

All LLM calls are mocked so tests run fully offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import os

import pytest

from src.agents.base_agent import Agent
from src.crew.dev_crew import DevCrew, _safe_filename


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_agent(role: str) -> Agent:
    llm = MagicMock()
    llm.chat.return_value = f"Response from {role}"
    return Agent(role=role, goal="Goal", backstory="Backstory", llm=llm)


@pytest.fixture()
def minimal_crew(tmp_path):
    """A crew with just a Product Manager agent, writing to a temp directory."""
    agent = _make_mock_agent("Product Manager")
    return DevCrew(
        agents=[agent],
        output_dir=tmp_path,
        save_individual=True,
        save_report=True,
    )


@pytest.fixture()
def full_crew(tmp_path):
    """A crew with all 5 main agents."""
    roles = [
        "Product Manager",
        "Software Architect",
        "Backend Developer",
        "QA Engineer",
        "Code Reviewer",
    ]
    agents = [_make_mock_agent(r) for r in roles]
    return DevCrew(
        agents=agents,
        output_dir=tmp_path,
        save_individual=True,
        save_report=True,
    )


# ---------------------------------------------------------------------------
# DevCrew.kickoff
# ---------------------------------------------------------------------------


def test_kickoff_returns_dict_with_all_roles(full_crew):
    outputs = full_crew.kickoff("Build a chat app", project_name="test_project")
    assert set(outputs.keys()) == {
        "Product Manager",
        "Software Architect",
        "Backend Developer",
        "QA Engineer",
        "Code Reviewer",
    }


def test_kickoff_responses_are_strings(full_crew):
    outputs = full_crew.kickoff("Build a chat app", project_name="test_project")
    for role, response in outputs.items():
        assert isinstance(response, str), f"{role} response is not a string"


def test_kickoff_each_agent_called_once(full_crew):
    full_crew.kickoff("Build a chat app", project_name="test_project")
    for agent in full_crew.agents:
        agent.llm.chat.assert_called_once()


def test_kickoff_later_agents_receive_context(full_crew):
    """Verify that each agent (except the first) receives prior output as context."""
    full_crew.kickoff("Build a chat app", project_name="test_project")

    for i, agent in enumerate(full_crew.agents):
        _, call_kwargs = agent.llm.chat.call_args
        # chat is called with positional args: system_prompt, user_message
        user_message = agent.llm.chat.call_args[0][1]
        if i == 0:
            # first agent should NOT have a prior context section
            assert "Context from previous team members" not in user_message
        else:
            # subsequent agents should see context
            assert "Context from previous team members" in user_message


def test_kickoff_saves_individual_files(full_crew, tmp_path):
    full_crew.kickoff("Build a chat app", project_name="test_project")
    md_files = list(tmp_path.rglob("*.md"))
    # 5 individual files + 1 final report
    assert len(md_files) >= 6


def test_kickoff_saves_final_report(full_crew, tmp_path):
    full_crew.kickoff("Build a chat app", project_name="test_project")
    reports = list(tmp_path.rglob("FINAL_REPORT.md"))
    assert len(reports) == 1
    content = reports[0].read_text()
    assert "Build a chat app" in content


def test_kickoff_no_save_when_disabled(tmp_path):
    agent = _make_mock_agent("Product Manager")
    crew = DevCrew(
        agents=[agent],
        output_dir=tmp_path,
        save_individual=False,
        save_report=False,
    )
    crew.kickoff("Test requirements", project_name="test_project")
    assert list(tmp_path.rglob("*.md")) == []


def test_kickoff_unknown_role_raises(tmp_path):
    agent = _make_mock_agent("Unknown Role")
    crew = DevCrew(agents=[agent], output_dir=tmp_path)
    with pytest.raises(ValueError, match="No task defined"):
        crew.kickoff("Some requirements", project_name="test_project")


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------


def test_safe_filename_replaces_spaces():
    assert " " not in _safe_filename("Product Manager")


def test_safe_filename_lowercases():
    assert _safe_filename("ProductManager") == "productmanager"


def test_safe_filename_keeps_alphanumeric_and_dash_underscore():
    assert _safe_filename("my-project_v2") == "my-project_v2"


def test_kickoff_runs_backend_fix_pass_when_major_findings(tmp_path):
    roles = [
        "Product Manager",
        "Software Architect",
        "Backend Developer",
        "QA Engineer",
        "Code Reviewer",
    ]
    agents = [_make_mock_agent(r) for r in roles]
    for agent in agents:
        if agent.role == "QA Engineer":
            agent.llm.chat.return_value = (
                "## Must-Address Checklist\n"
                "- [Major] Add validation for empty payload.\n"
            )
        elif agent.role == "Code Reviewer":
            agent.llm.chat.return_value = (
                "## Must-Address Checklist\n"
                "- [Minor] Improve naming consistency.\n"
            )
    crew = DevCrew(
        agents=agents,
        output_dir=tmp_path,
        max_fix_iterations=1,
        stop_on_no_major_issues=True,
    )
    crew.kickoff("Build API", project_name="api_project")

    backend = next(a for a in agents if a.role == "Backend Developer")
    assert backend.llm.chat.call_count == 2
    second_call_user_message = backend.llm.chat.call_args_list[1][0][1]
    assert "Must-Address Checklist" in second_call_user_message
