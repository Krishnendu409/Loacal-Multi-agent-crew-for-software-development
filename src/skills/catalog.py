"""Skills catalog – curated operational skills for each agent role.

BUG FIX: Reduced skill descriptions to be concise single-line entries.
Verbose multi-paragraph skill blocks were overflowing context windows on
phi3:mini and causing JSON output failures.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Shared skills injected into every agent
# ---------------------------------------------------------------------------

SHARED_SKILLS: dict[str, str] = {
    "structured communication": (
        "Structure outputs clearly: summary first, then details, then action items."
    ),
    "security-first thinking": (
        "Default to least-privilege; validate all inputs; never expose secrets in logs."
    ),
    "verification mindset": (
        "State assumptions explicitly; flag unresolved ambiguities rather than guessing."
    ),
    "documentation discipline": (
        "Every decision needs a rationale; every interface needs a contract."
    ),
}

# ---------------------------------------------------------------------------
# Role-specific skills
# ---------------------------------------------------------------------------

ROLE_SKILLS: dict[str, dict[str, str]] = {
    "ceo_planner": {
        "strategic framing": "Convert vague goals into measurable outcomes and phased plans.",
        "risk-first planning": "Identify blockers before committing to execution paths.",
    },
    "market_researcher": {
        "evidence-based analysis": "Ground recommendations in observable market signals.",
        "competitive differentiation": "Map competitor weaknesses to product opportunities.",
    },
    "customer_support_feedback_analyst": {
        "support signal prioritization": "Rank pain points by frequency × severity × fix cost.",
        "feedback loop design": "Identify what telemetry and support channels are needed.",
    },
    "product_manager": {
        "requirement precision": "Turn ambiguous wishes into testable acceptance criteria.",
        "scope discipline": "Explicitly separate MVP from future iterations.",
    },
    "compliance_privacy_specialist": {
        "privacy impact thinking": "Apply data minimisation and consent requirements early.",
        "regulatory mapping": "Link requirements to applicable standards (GDPR, SOC2, etc).",
    },
    "architect": {
        "simplicity bias": "Choose the simplest architecture that meets requirements.",
        "trade-off documentation": "Record why rejected options were rejected.",
    },
    "ui_ux_designer": {
        "interaction design and prototype clarity": (
            "Define user journeys and interaction states before visual design."
        ),
        "accessibility first": "Meet WCAG 2.1 AA; never bolt on accessibility after the fact.",
    },
    "database_engineer": {
        "schema evolution safety": "Every schema change must be reversible or zero-downtime.",
        "index discipline": "Index for query patterns; avoid over-indexing.",
    },
    "api_integration_engineer": {
        "resilient integration contracts": (
            "Design for retry, circuit-breaker, and idempotency from day one."
        ),
        "contract-first design": "Define the API contract before writing implementation.",
    },
    "frontend_developer": {
        "ui consistency and accessibility": (
            "Follow the design system; meet accessibility requirements."
        ),
        "bundle hygiene": "Avoid heavy dependencies; measure bundle size impact.",
    },
    "backend_developer": {
        "dependency hygiene": "Pin versions; audit for vulnerabilities; prefer standard library.",
        "error boundary design": "Every external call must have explicit failure handling.",
    },
    "data_analytics_engineer": {
        "measurement reliability": "Define events and metrics before instrumenting.",
        "schema-on-write discipline": "Validate data at ingestion; fail fast on bad data.",
    },
    "performance_engineer": {
        "evidence-driven optimization": "Profile before optimizing; cite measurements.",
        "latency budget": "Assign latency targets per layer; track cumulative budget.",
    },
    "security_engineer": {
        "threat-led validation": "Model attacker goals first; then identify controls.",
        "remediation specificity": "Provide concrete fix guidance, not just vulnerability names.",
    },
    "qa_engineer": {
        "risk-based test prioritization": "Test highest-risk paths first; document coverage gaps.",
        "test isolation": "Each test must be independent and deterministic.",
    },
    "code_reviewer": {
        "constructive specificity": "Cite file/line; explain why; suggest a concrete fix.",
        "severity tagging": "Tag every finding [Critical] / [Major] / [Minor].",
    },
    "technical_writer": {
        "high-signal operational documentation": (
            "Write for the 3am incident responder; every runbook must be actionable."
        ),
        "audience-appropriate depth": "Developer docs vs user docs vs ops runbooks differ.",
    },
    "sre_reliability_engineer": {
        "slo-first reliability planning": "Define SLOs before designing monitoring.",
        "toil reduction": "Automate anything that is repetitive and manual.",
    },
    "release_manager": {
        "go-no-go discipline": "Every release needs explicit gates and a rollback plan.",
        "staged rollout": "Never release to 100% at once; use canary or feature flags.",
    },
    "devops_engineer": {
        "infrastructure-as-code": "No manual infrastructure changes; everything versioned.",
        "pipeline security": "Secrets in vault; least-privilege service accounts.",
    },
}


def get_skills_for_role(
    role_key: str,
    shared_keys: list[str] | None = None,
    extra_keys: list[str] | None = None,
    exclude_keys: list[str] | None = None,
    max_skills: int = 8,
) -> list[str]:
    """Return a list of skill description strings for *role_key*.

    Skills are selected in priority order:
      1. Role-specific skills from ROLE_SKILLS
      2. Shared skills from SHARED_SKILLS (filtered by *shared_keys*)

    The result is capped at *max_skills* to prevent context overflow.
    """
    skills: list[str] = []
    exclude = set(exclude_keys or [])

    # Role-specific first
    role_skill_map = ROLE_SKILLS.get(role_key, {})
    for name, desc in role_skill_map.items():
        if name not in exclude:
            skills.append(desc)

    # Then shared skills
    keys_to_include = shared_keys if shared_keys is not None else list(SHARED_SKILLS.keys())
    for name in keys_to_include:
        if name in exclude:
            continue
        desc = SHARED_SKILLS.get(name)
        if desc:
            skills.append(desc)

    # Optional extras
    if extra_keys:
        for name in extra_keys:
            desc = SHARED_SKILLS.get(name) or role_skill_map.get(name)
            if desc and desc not in skills:
                skills.append(desc)

    return skills[:max_skills]


def resolve_agent_skills(
    role_key: str,
    skills_config: dict[str, Any] | None,
) -> list[str]:
    """Resolve skills for *role_key* using the skills configuration block.

    Returns an empty list if skills_config is None or disabled.
    """
    if not skills_config:
        return []

    if not skills_config.get("include_default_role_skills", True):
        return []

    max_skills = int(skills_config.get("max_skills_per_agent", 8))
    shared_keys: list[str] = list(skills_config.get("shared", SHARED_SKILLS.keys()))

    # Per-role overrides from config
    per_role: dict[str, list[str]] = {}
    raw_per_role = skills_config.get("per_role", {})
    if isinstance(raw_per_role, dict):
        per_role = {k: list(v) for k, v in raw_per_role.items() if isinstance(v, list)}

    extra_keys: list[str] = list(per_role.get(role_key, []))

    # Global include / exclude lists
    include: list[str] = list(skills_config.get("include", []))
    exclude: list[str] = list(skills_config.get("exclude", []))

    # Per-role include / exclude
    per_role_include: dict[str, list[str]] = {}
    raw_pri = skills_config.get("per_role_include", {})
    if isinstance(raw_pri, dict):
        per_role_include = {k: list(v) for k, v in raw_pri.items() if isinstance(v, list)}
    extra_keys.extend(per_role_include.get(role_key, []))
    extra_keys.extend(include)

    per_role_exclude: dict[str, list[str]] = {}
    raw_pre = skills_config.get("per_role_exclude", {})
    if isinstance(raw_pre, dict):
        per_role_exclude = {k: list(v) for k, v in raw_pre.items() if isinstance(v, list)}
    exclude.extend(per_role_exclude.get(role_key, []))

    return get_skills_for_role(
        role_key=role_key,
        shared_keys=shared_keys,
        extra_keys=extra_keys,
        exclude_keys=exclude,
        max_skills=max_skills,
    )
