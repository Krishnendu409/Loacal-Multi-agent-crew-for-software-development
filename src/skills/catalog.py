"""Built-in skill catalog for software-team agent roles."""

from __future__ import annotations

from src.skills.ecc_pack import ecc_conflicts_map, ecc_priority_map, resolve_ecc_pack_labels


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


def _extract_list(config: dict[str, object] | None, key: str) -> list[object]:
    if not config:
        return []
    value = config.get(key, [])
    if isinstance(value, list):
        return value
    return []


def _extract_role_list(
    config: dict[str, object] | None, key: str, role_key: str
) -> list[object]:
    if not config:
        return []
    value = config.get(key, {})
    if not isinstance(value, dict):
        return []
    role_items = value.get(role_key, [])
    if isinstance(role_items, list):
        return role_items
    return []


def _resolve_external_pack_skills(role_key: str, skills_config: dict[str, object] | None) -> list[str]:
    if not skills_config:
        return []
    packs = skills_config.get("packs", {})
    if not isinstance(packs, dict):
        return []
    ecc_cfg = packs.get("ecc", {})
    if not isinstance(ecc_cfg, dict):
        return []
    if not bool(ecc_cfg.get("enabled", False)):
        return []
    profile = str(ecc_cfg.get("profile", "starter")).strip() or "starter"
    return resolve_ecc_pack_labels(profile, role_key)


def _apply_exclusions(
    role_key: str, skills: list[str], skills_config: dict[str, object] | None
) -> list[str]:
    if not skills_config:
        return skills
    excluded = set(_clean(_extract_list(skills_config, "exclude")))
    excluded.update(_clean(_extract_role_list(skills_config, "per_role_exclude", role_key)))
    if not excluded:
        return skills
    lowered = {x.lower() for x in excluded}
    return [skill for skill in skills if skill.lower() not in lowered]


def _apply_conflict_guardrails(skills: list[str], priority_map: dict[str, int]) -> list[str]:
    conflicts = ecc_conflicts_map()
    if not conflicts:
        return skills
    kept: list[str] = []
    for skill in skills:
        blockers = conflicts.get(skill, set())
        if not blockers:
            kept.append(skill)
            continue
        current_p = priority_map.get(skill, 3)
        conflicting_existing = [existing for existing in kept if existing in blockers]
        if not conflicting_existing:
            kept.append(skill)
            continue
        # Lower-or-equal priority keeps the existing earlier skill for deterministic output.
        if any(priority_map.get(existing, 3) <= current_p for existing in conflicting_existing):
            continue
        for existing in conflicting_existing:
            kept.remove(existing)
        kept.append(skill)
    return kept


def _apply_budget(skills: list[str], skills_config: dict[str, object] | None) -> list[str]:
    if not skills_config:
        return skills
    raw = skills_config.get("max_skills_per_agent")
    if raw is None:
        return skills
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        return skills
    if budget <= 0 or len(skills) <= budget:
        return skills
    priority_map = ecc_priority_map()
    scored = sorted(
        (priority_map.get(skill, 3), idx, skill) for idx, skill in enumerate(skills)
    )
    selected = {skill for _, _, skill in scored[:budget]}
    return [skill for skill in skills if skill in selected]


def _validate_strict_config(role_key: str, skills_config: dict[str, object] | None) -> None:
    if not skills_config or not bool(skills_config.get("strict_mode", False)):
        return

    known = set()
    for values in ROLE_DEFAULT_SKILLS.values():
        known.update(v.lower() for v in values)
    known.update(label.lower() for label in ecc_priority_map().keys())
    known.update(v.lower() for v in _clean(_extract_list(skills_config, "shared")))
    for role in ROLE_DEFAULT_SKILLS:
        known.update(v.lower() for v in _clean(_extract_role_list(skills_config, "per_role", role)))
        known.update(
            v.lower() for v in _clean(_extract_role_list(skills_config, "per_role_include", role))
        )

    candidates = _clean(_extract_list(skills_config, "exclude"))
    candidates.extend(_clean(_extract_role_list(skills_config, "per_role_exclude", role_key)))
    candidates.extend(_clean(_extract_list(skills_config, "include")))
    candidates.extend(_clean(_extract_role_list(skills_config, "per_role_include", role_key)))
    for item in candidates:
        if item.lower() not in known:
            raise ValueError(
                f"Unknown skill reference in strict_mode for role '{role_key}': '{item}'"
            )


def resolve_agent_skills(role_key: str, skills_config: dict[str, object] | None) -> list[str]:
    """Resolve final skills for a role from defaults + external packs + config overlays."""
    include_defaults = True
    if skills_config and "include_default_role_skills" in skills_config:
        include_defaults = bool(skills_config.get("include_default_role_skills"))

    _validate_strict_config(role_key, skills_config)

    merged: list[object] = []
    if include_defaults:
        merged.extend(ROLE_DEFAULT_SKILLS.get(role_key, []))

    merged.extend(_resolve_external_pack_skills(role_key, skills_config))
    merged.extend(_extract_list(skills_config, "shared"))
    merged.extend(_extract_role_list(skills_config, "per_role", role_key))
    merged.extend(_extract_list(skills_config, "include"))
    merged.extend(_extract_role_list(skills_config, "per_role_include", role_key))

    resolved = _clean(merged)
    resolved = _apply_exclusions(role_key, resolved, skills_config)
    resolved = _apply_conflict_guardrails(resolved, ecc_priority_map())
    resolved = _apply_budget(resolved, skills_config)
    return resolved
