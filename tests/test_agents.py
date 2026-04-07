"""Unit tests for the Agent base class and agent definitions.

All LLM calls are mocked so these tests run offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.base_agent import Agent
from src.agents.definitions import build_agents, AGENT_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm():
    """Return a mock OllamaClient that echoes the user message back."""
    llm = MagicMock()
    llm.chat.side_effect = lambda system, user, **kwargs: f"[MOCK RESPONSE] {user[:80]}"
    return llm


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------


def test_agent_execute_calls_llm(mock_llm):
    agent = Agent(
        role="Product Manager",
        goal="Write specs",
        backstory="Experienced PM",
        llm=mock_llm,
    )
    result = agent.execute("Build a todo app")
    mock_llm.chat.assert_called_once()
    assert isinstance(result, str)
    assert len(result) > 0


def test_agent_execute_includes_context_in_user_message(mock_llm):
    agent = Agent(
        role="Architect",
        goal="Design systems",
        backstory="Experienced Architect",
        llm=mock_llm,
    )
    agent.execute("Design the API", context="## Product Manager\n\nSpec content here")
    _, user_message = mock_llm.chat.call_args[0]
    assert "Context from previous team members" in user_message
    assert "Spec content here" in user_message


def test_agent_system_prompt_contains_role(mock_llm):
    agent = Agent(
        role="QA Engineer",
        goal="Test everything",
        backstory="Detail-oriented tester",
        llm=mock_llm,
    )
    agent.execute("Write tests")
    system_prompt, _ = mock_llm.chat.call_args[0]
    assert "QA Engineer" in system_prompt
    assert "Test everything" in system_prompt


def test_agent_extra_instructions_in_system_prompt(mock_llm):
    agent = Agent(
        role="Developer",
        goal="Code stuff",
        backstory="Writes code",
        llm=mock_llm,
        extra_instructions="Always use type hints.",
    )
    agent.execute("Implement feature X")
    system_prompt, _ = mock_llm.chat.call_args[0]
    assert "Always use type hints." in system_prompt


def test_agent_system_prompt_includes_skills_and_handoff_contract(mock_llm):
    agent = Agent(
        role="Developer",
        goal="Build features",
        backstory="Writes robust code",
        llm=mock_llm,
        skills=["secure coding", "structured communication"],
        enforce_handoff_sections=True,
    )
    agent.execute("Implement feature Y")
    system_prompt, _ = mock_llm.chat.call_args[0]
    assert "Apply these operational skills" in system_prompt
    assert "secure coding" in system_prompt
    assert "Handoff Notes for Next Role" in system_prompt


# ---------------------------------------------------------------------------
# build_agents
# ---------------------------------------------------------------------------


def test_build_agents_default_excludes_devops(mock_llm):
    agents = build_agents(mock_llm)
    roles = [a.role for a in agents]
    assert "DevOps Engineer" not in roles
    assert "CEO Planner" in roles
    assert "Market Researcher" in roles
    assert "Product Manager" in roles


def test_build_agents_enabled_all(mock_llm):
    enabled = {k: True for k in AGENT_ORDER}
    agents = build_agents(mock_llm, enabled=enabled)
    assert len(agents) == len(AGENT_ORDER)


def test_build_agents_only_pm_and_dev(mock_llm):
    enabled = {k: False for k in AGENT_ORDER}
    enabled["product_manager"] = True
    enabled["backend_developer"] = True
    agents = build_agents(mock_llm, enabled=enabled)
    roles = [a.role for a in agents]
    assert roles == ["Product Manager", "Backend Developer"]


def test_build_agents_preserves_order(mock_llm):
    enabled = {k: True for k in AGENT_ORDER}
    agents = build_agents(mock_llm, enabled=enabled)
    expected_order = [
        "CEO Planner",
        "Market Researcher",
        "Product Manager",
        "Software Architect",
        "Frontend Developer",
        "Backend Developer",
        "QA Engineer",
        "Code Reviewer",
        "DevOps Engineer",
    ]
    assert [a.role for a in agents] == expected_order


def test_build_agents_empty_when_all_disabled(mock_llm):
    enabled = {k: False for k in AGENT_ORDER}
    agents = build_agents(mock_llm, enabled=enabled)
    assert agents == []


def test_build_agents_applies_role_routing_and_fallbacks(mock_llm):
    enabled = {k: False for k in AGENT_ORDER}
    enabled["backend_developer"] = True
    llm_config = {
        "routing": {"backend_developer": "deepseek-coder:6.7b"},
        "fallbacks": {"backend_developer": ["qwen2.5:7b-instruct", "phi3:mini"]},
        "role_options": {"backend_developer": {"num_predict": 1024, "temperature": 0.2}},
    }
    agents = build_agents(mock_llm, enabled=enabled, llm_config=llm_config)
    assert len(agents) == 1
    agent = agents[0]
    assert agent.llm_model == "deepseek-coder:6.7b"
    assert agent.llm_fallback_models == ["qwen2.5:7b-instruct", "phi3:mini"]
    assert agent.llm_options == {"num_predict": 1024, "temperature": 0.2}


def test_build_agents_filters_models_against_allow_list(mock_llm):
    enabled = {k: False for k in AGENT_ORDER}
    enabled["backend_developer"] = True
    llm_config = {
        "allowed_models": ["qwen2.5:7b-instruct", "deepseek-coder:6.7b", "phi3:mini"],
        "routing": {"backend_developer": "llama3.2:3b"},
        "fallbacks": {"backend_developer": ["deepseek-coder:6.7b", "mistral"]},
    }
    agents = build_agents(mock_llm, enabled=enabled, llm_config=llm_config)
    assert len(agents) == 1
    agent = agents[0]
    assert agent.llm_model is None
    assert agent.llm_fallback_models == ["deepseek-coder:6.7b"]


def test_build_agents_applies_skill_config(mock_llm):
    enabled = {k: False for k in AGENT_ORDER}
    enabled["backend_developer"] = True
    skills_config = {
        "shared": ["security-first thinking"],
        "per_role": {"backend_developer": ["dependency hygiene"]},
        "include_default_role_skills": True,
        "enforce_handoff_sections": False,
    }
    agents = build_agents(
        mock_llm,
        enabled=enabled,
        llm_config={},
        skills_config=skills_config,
    )
    assert len(agents) == 1
    agent = agents[0]
    assert "dependency hygiene" in agent.skills
    assert "security-first thinking" in agent.skills
    assert agent.enforce_handoff_sections is False


def test_build_agents_applies_external_ecc_pack(mock_llm):
    enabled = {k: False for k in AGENT_ORDER}
    enabled["qa_engineer"] = True
    agents = build_agents(
        mock_llm,
        enabled=enabled,
        llm_config={},
        skills_config={
            "packs": {"ecc": {"enabled": True, "profile": "starter"}},
            "include_default_role_skills": False,
        },
    )
    assert len(agents) == 1
    agent = agents[0]
    assert "python-testing" in agent.skills
