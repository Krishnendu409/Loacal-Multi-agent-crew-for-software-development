"""Unit tests for skill catalog resolution."""

from __future__ import annotations

from src.skills.catalog import resolve_agent_skills


def test_resolve_agent_skills_uses_defaults_for_role():
    skills = resolve_agent_skills("backend_developer", None)
    assert "secure coding" in skills


def test_resolve_agent_skills_merges_shared_and_per_role():
    skills = resolve_agent_skills(
        "qa_engineer",
        {
            "shared": ["structured communication"],
            "per_role": {"qa_engineer": ["risk-based test prioritization"]},
        },
    )
    assert "structured communication" in skills
    assert "risk-based test prioritization" in skills


def test_resolve_agent_skills_can_disable_defaults():
    skills = resolve_agent_skills(
        "architect",
        {
            "include_default_role_skills": False,
            "shared": ["structured communication"],
            "per_role": {"architect": ["domain-driven design"]},
        },
    )
    assert skills == ["structured communication", "domain-driven design"]
