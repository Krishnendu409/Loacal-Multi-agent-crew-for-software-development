"""Unit tests for skill catalog resolution."""

from __future__ import annotations

import pytest

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


def test_resolve_agent_skills_includes_ecc_pack_when_enabled():
    skills = resolve_agent_skills(
        "backend_developer",
        {
            "packs": {"ecc": {"enabled": True, "profile": "starter"}},
        },
    )
    assert "python-patterns" in skills
    assert "tdd-workflow" in skills


def test_resolve_agent_skills_supports_exclusions():
    skills = resolve_agent_skills(
        "backend_developer",
        {
            "packs": {"ecc": {"enabled": True, "profile": "starter"}},
            "exclude": ["python-patterns"],
            "per_role_exclude": {"backend_developer": ["tdd-workflow"]},
        },
    )
    assert "python-patterns" not in skills
    assert "tdd-workflow" not in skills


def test_resolve_agent_skills_applies_budget_by_priority():
    skills = resolve_agent_skills(
        "backend_developer",
        {
            "include_default_role_skills": False,
            "packs": {"ecc": {"enabled": True, "profile": "starter"}},
            "max_skills_per_agent": 2,
        },
    )
    assert len(skills) == 2
    assert "search-first" in skills
    assert "context-budget" in skills


def test_resolve_agent_skills_strict_mode_rejects_unknown_references():
    with pytest.raises(ValueError, match="Unknown skill reference"):
        resolve_agent_skills(
            "backend_developer",
            {
                "strict_mode": True,
                "exclude": ["not-a-real-skill"],
            },
        )
