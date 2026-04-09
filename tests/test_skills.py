"""Unit tests for skill catalog resolution."""

from __future__ import annotations

import pytest

from src.skills.catalog import resolve_agent_skills


def test_resolve_agent_skills_uses_defaults_for_role():
    skills = resolve_agent_skills("backend_developer", None)
    assert "secure coding" in skills


def test_resolve_agent_skills_uses_defaults_for_uiux_and_security_roles():
    uiux_skills = resolve_agent_skills("ui_ux_designer", None)
    sec_skills = resolve_agent_skills("security_engineer", None)
    assert "interaction design" in uiux_skills
    assert "threat modeling" in sec_skills


def test_resolve_agent_skills_uses_defaults_for_new_specialist_roles():
    db_skills = resolve_agent_skills("database_engineer", None)
    perf_skills = resolve_agent_skills("performance_engineer", None)
    writer_skills = resolve_agent_skills("technical_writer", None)
    assert "schema design and normalization" in db_skills
    assert "performance bottleneck analysis" in perf_skills
    assert "runbook authoring" in writer_skills


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


def test_resolve_agent_skills_includes_ecc_pack_for_uiux_and_security_roles():
    uiux_skills = resolve_agent_skills(
        "ui_ux_designer",
        {
            "include_default_role_skills": False,
            "packs": {"ecc": {"enabled": True, "profile": "starter"}},
        },
    )
    sec_skills = resolve_agent_skills(
        "security_engineer",
        {
            "include_default_role_skills": False,
            "packs": {"ecc": {"enabled": True, "profile": "starter"}},
        },
    )
    assert "ux-design" in uiux_skills
    assert "security-review" in sec_skills


def test_resolve_agent_skills_includes_ecc_pack_for_new_specialist_roles():
    api_skills = resolve_agent_skills(
        "api_integration_engineer",
        {
            "include_default_role_skills": False,
            "packs": {"ecc": {"enabled": True, "profile": "starter"}},
        },
    )
    compliance_skills = resolve_agent_skills(
        "compliance_privacy_specialist",
        {
            "include_default_role_skills": False,
            "packs": {"ecc": {"enabled": True, "profile": "starter"}},
        },
    )
    assert "api-design" in api_skills
    assert "security-review" in compliance_skills


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


def test_resolve_agent_skills_strict_mode_rejects_unknown_includes():
    with pytest.raises(ValueError, match="Unknown skill reference"):
        resolve_agent_skills(
            "backend_developer",
            {
                "strict_mode": True,
                "include": ["unknown-include-skill"],
            },
        )
