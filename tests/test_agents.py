"""Unit tests for the Agent base class and agent definitions.

All LLM calls are mocked so these tests run offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.base_agent import Agent
from src.agents.definitions import AGENT_ORDER, build_agents, register_agent_role
from src.models.schemas import ArchitectHandoffSchema


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
    assert "Customer Support/Feedback Analyst" in roles
    assert "Compliance & Privacy Specialist" in roles
    assert "UI/UX Designer" in roles
    assert "Database Engineer" in roles
    assert "API Integration Engineer" in roles
    assert "Security Engineer" in roles
    assert "Performance Engineer" in roles
    assert "Technical Writer" in roles
    assert "SRE / Reliability Engineer" in roles
    assert "Release Manager" in roles
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
        "Customer Support/Feedback Analyst",
        "Product Manager",
        "Compliance & Privacy Specialist",
        "Software Architect",
        "UI/UX Designer",
        "Database Engineer",
        "API Integration Engineer",
        "Frontend Developer",
        "Backend Developer",
        "Data/Analytics Engineer",
        "Performance Engineer",
        "Security Engineer",
        "QA Engineer",
        "Code Reviewer",
        "Technical Writer",
        "SRE / Reliability Engineer",
        "Release Manager",
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
        "role_retries": {"backend_developer": 3},
    }
    agents = build_agents(mock_llm, enabled=enabled, llm_config=llm_config)
    assert len(agents) == 1
    agent = agents[0]
    assert agent.llm_model == "deepseek-coder:6.7b"
    assert agent.llm_fallback_models == ["qwen2.5:7b-instruct", "phi3:mini"]
    assert agent.llm_options == {"num_predict": 1024, "temperature": 0.2}
    assert agent.llm_retries == 3


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


# ---------------------------------------------------------------------------
# P0 #3: no silent TypeError swallowing in Agent.execute
# ---------------------------------------------------------------------------


def test_agent_execute_propagates_runtime_error():
    """If the LLM raises an error other than TypeError it should propagate."""
    llm = MagicMock()
    llm.chat.side_effect = RuntimeError("connection refused")
    agent = Agent(role="Product Manager", goal="Write specs", backstory="PM", llm=llm)
    with pytest.raises(RuntimeError, match="connection refused"):
        agent.execute("Build a todo app")


def test_agent_execute_passes_model_and_options_to_llm():
    """Agent.execute must forward llm_model, llm_options, and llm_fallback_models."""
    llm = MagicMock()
    llm.chat.return_value = "ok"
    agent = Agent(
        role="Backend Developer",
        goal="Write code",
        backstory="Senior dev",
        llm=llm,
        llm_model="deepseek-coder:6.7b",
        llm_options={"temperature": 0.2},
        llm_fallback_models=["qwen2.5:7b-instruct"],
        llm_retries=2,
    )
    agent.execute("Implement feature")
    _, kwargs = llm.chat.call_args
    assert kwargs.get("model") == "deepseek-coder:6.7b"
    assert kwargs.get("options") == {"temperature": 0.2}
    assert kwargs.get("fallback_models") == ["qwen2.5:7b-instruct"]
    assert kwargs.get("retries_override") == 2


def test_agent_execute_does_not_swallow_type_error():
    """A TypeError from the LLM (e.g., unexpected keyword argument) must propagate."""
    llm = MagicMock()
    llm.chat.side_effect = TypeError("unexpected keyword argument 'model'")
    agent = Agent(role="Product Manager", goal="Specs", backstory="PM", llm=llm)
    with pytest.raises(TypeError):
        agent.execute("Build feature")


def test_agent_execute_passes_format_schema_when_output_schema_set():
    llm = MagicMock()
    llm.chat.return_value = "{}"
    agent = Agent(
        role="Software Architect",
        goal="Design architecture",
        backstory="Architect",
        llm=llm,
        output_schema=ArchitectHandoffSchema,
    )
    agent.execute("Design system")
    _, kwargs = llm.chat.call_args
    assert isinstance(kwargs.get("format_schema"), dict)
    assert "properties" in kwargs["format_schema"]


# ---------------------------------------------------------------------------
# P0 #1: OllamaClient reuses the same underlying client instance
# ---------------------------------------------------------------------------


def test_ollama_client_reuses_instance():
    """The same ollama.Client object must be returned on every call."""
    from unittest.mock import MagicMock, patch
    from src.utils.ollama_client import OllamaClient

    mock_ollama_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_ollama_module.Client.return_value = mock_client_instance

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini")
        c1 = oc._get_client()
        c2 = oc._get_client()

    assert c1 is c2
    # Client constructor called exactly once despite two _get_client() calls
    mock_ollama_module.Client.assert_called_once()


def test_ollama_client_chat_reuses_instance_across_calls():
    """Calling chat() multiple times must not create a new ollama.Client each time."""
    from unittest.mock import MagicMock, patch
    from src.utils.ollama_client import OllamaClient

    mock_ollama_module = MagicMock()
    mock_client_instance = MagicMock()
    # Simulate a successful chat response
    mock_client_instance.chat.return_value = {"message": {"content": "hello"}}
    mock_ollama_module.Client.return_value = mock_client_instance

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini")
        oc.chat("system", "user message 1")
        oc.chat("system", "user message 2")

    # The underlying ollama.Client constructor must have been called only once
    mock_ollama_module.Client.assert_called_once()


def test_ollama_client_chat_raises_on_missing_message_content():
    """Malformed responses must raise a clear runtime error."""
    from unittest.mock import MagicMock, patch

    from src.utils.ollama_client import OllamaClient

    mock_ollama_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.chat.return_value = {"unexpected": "shape"}
    mock_ollama_module.Client.return_value = mock_client_instance

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini")
        with pytest.raises(RuntimeError, match="missing message.content"):
            oc.chat("system", "user")


def test_ollama_client_chat_accepts_legacy_response_shape():
    """Legacy Ollama responses with top-level `response` should be accepted."""
    from unittest.mock import MagicMock, patch

    from src.utils.ollama_client import OllamaClient

    mock_ollama_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.chat.return_value = {"response": "legacy text"}
    mock_ollama_module.Client.return_value = mock_client_instance

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini")
        assert oc.chat("system", "user") == "legacy text"


def test_ollama_client_chat_accepts_object_response_shape():
    """Ollama SDK object responses should be accepted by normalizing model_dump()."""
    from unittest.mock import MagicMock, patch

    from src.utils.ollama_client import OllamaClient

    class _MockChatResponse:
        def model_dump(self) -> dict[str, object]:
            return {"message": {"content": "object text"}}

    mock_ollama_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.chat.return_value = _MockChatResponse()
    mock_ollama_module.Client.return_value = mock_client_instance

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini")
        assert oc.chat("system", "user") == "object text"


def test_ollama_client_chat_accepts_list_content_shape():
    """List-based multimodal content should be reduced to concatenated text."""
    from unittest.mock import MagicMock, patch

    from src.utils.ollama_client import OllamaClient

    mock_ollama_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.chat.return_value = {
        "message": {
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"},
            ]
        }
    }
    mock_ollama_module.Client.return_value = mock_client_instance

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini")
        assert oc.chat("system", "user") == "hello world"


def test_ollama_client_retries_timeout_with_extended_timeout():
    """Timeout errors should trigger a retry with a larger timeout budget."""
    from unittest.mock import MagicMock, patch

    from src.utils.ollama_client import OllamaClient

    mock_ollama_module = MagicMock()
    first_client = MagicMock()
    second_client = MagicMock()
    first_client.chat.side_effect = TimeoutError("timed out")
    second_client.chat.return_value = {"message": {"content": "recovered"}}
    mock_ollama_module.Client.side_effect = [first_client, second_client]

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini", timeout_seconds=120, retries=0)
        response = oc.chat("system", "user")

    assert response == "recovered"
    assert mock_ollama_module.Client.call_count == 2
    assert mock_ollama_module.Client.call_args_list[0].kwargs["timeout"] == 120
    assert mock_ollama_module.Client.call_args_list[1].kwargs["timeout"] == 240


def test_ollama_client_cache_includes_options():
    """Different options must not reuse the same cached response."""
    from unittest.mock import MagicMock, patch

    from src.utils.ollama_client import OllamaClient

    mock_ollama_module = MagicMock()
    mock_client_instance = MagicMock()

    def _chat(**kwargs):
        temperature = (kwargs.get("options") or {}).get("temperature")
        return {"message": {"content": f"temp={temperature}"}}

    mock_client_instance.chat.side_effect = _chat
    mock_ollama_module.Client.return_value = mock_client_instance

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini")
        first = oc.chat("system", "user", options={"temperature": 0.1})
        second = oc.chat("system", "user", options={"temperature": 0.2})

    assert first == "temp=0.1"
    assert second == "temp=0.2"
    assert mock_client_instance.chat.call_count == 2


def test_ollama_client_cache_is_bounded():
    """Cache growth should be bounded to avoid unbounded memory usage."""
    from unittest.mock import MagicMock, patch

    from src.utils.ollama_client import OllamaClient

    mock_ollama_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.chat.return_value = {"message": {"content": "ok"}}
    mock_ollama_module.Client.return_value = mock_client_instance

    with patch("src.utils.ollama_client._get_ollama", return_value=mock_ollama_module):
        oc = OllamaClient(model="phi3:mini", max_cache_entries=2)
        oc.chat("system", "user-1")
        oc.chat("system", "user-2")
        oc.chat("system", "user-3")

    assert len(oc._cache) == 2


def test_register_agent_role_allows_plugin_extension(mock_llm):
    import src.agents.definitions as agent_defs

    key = "temporary_plugin_role"

    def _factory(llm, llm_config):
        return Agent(
            role="Temporary Plugin Role",
            goal="Test plugin path",
            backstory="Plugin agent",
            llm=llm,
        )

    register_agent_role(key, _factory, after="release_manager")
    try:
        enabled = {k: False for k in AGENT_ORDER}
        enabled[key] = True
        agents = build_agents(mock_llm, enabled=enabled)
        assert len(agents) == 1
        assert agents[0].role == "Temporary Plugin Role"
    finally:
        # Cleanup registry mutation for test isolation
        if key in AGENT_ORDER:
            AGENT_ORDER.remove(key)
        agent_defs._AGENT_FACTORIES.pop(key, None)
