"""Built-in skill catalog for software-team agent roles."""

from __future__ import annotations


ROLE_DEFAULT_SKILLS: dict[str, list[str]] = {
    "product_manager": [
        "requirements decomposition",
        "scope management",
        "acceptance criteria design",
        "risk identification",
    ],
    "architect": [
        "architecture trade-off analysis",
        "api and data contract design",
        "scalability and reliability planning",
        "threat modeling",
    ],
    "backend_developer": [
        "clean code and refactoring",
        "defensive programming",
        "secure coding",
        "testability-first implementation",
    ],
    "qa_engineer": [
        "test strategy design",
        "edge-case exploration",
        "regression prevention",
        "quality risk assessment",
    ],
    "code_reviewer": [
        "code quality review",
        "security review",
        "performance review",
        "actionable feedback writing",
    ],
    "devops_engineer": [
        "ci/cd design",
        "containerization",
        "release management",
        "observability planning",
    ],
}


def _clean(items: list[object]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


def resolve_agent_skills(role_key: str, skills_config: dict[str, object] | None) -> list[str]:
    """Resolve final skills for a role from defaults + config overlays."""
    include_defaults = True
    if skills_config and "include_default_role_skills" in skills_config:
        include_defaults = bool(skills_config.get("include_default_role_skills"))

    merged: list[object] = []
    if include_defaults:
        merged.extend(ROLE_DEFAULT_SKILLS.get(role_key, []))

    if not skills_config:
        return _clean(merged)

    shared = skills_config.get("shared", [])
    if isinstance(shared, list):
        merged.extend(shared)

    per_role = skills_config.get("per_role", {})
    if isinstance(per_role, dict):
        role_items = per_role.get(role_key, [])
        if isinstance(role_items, list):
            merged.extend(role_items)

    return _clean(merged)
