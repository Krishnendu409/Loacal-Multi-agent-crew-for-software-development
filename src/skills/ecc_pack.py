"""External ECC skill-pack registry and mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalSkill:
    key: str
    label: str
    description: str
    category: str
    recommended_roles: tuple[str, ...]
    priority: int
    conflict_with: tuple[str, ...] = ()


SELECTION_CRITERIA: tuple[str, ...] = (
    "high impact on output quality",
    "broad reuse across roles",
    "low conflict with existing instructions",
    "clear role relevance",
    "manageable prompt footprint",
)


ECC_CANONICAL_SKILLS: dict[str, CanonicalSkill] = {
    "api_design": CanonicalSkill(
        key="api_design",
        label="api-design",
        description="Design clear API contracts with versioning, validation, and failure semantics.",
        category="architecture",
        recommended_roles=("architect", "backend_developer", "code_reviewer"),
        priority=1,
    ),
    "strategy_planning": CanonicalSkill(
        key="strategy_planning",
        label="strategy-planning",
        description="Break complex goals into phased strategy with explicit decision gates.",
        category="strategy",
        recommended_roles=("ceo_planner", "product_manager", "architect"),
        priority=1,
    ),
    "market_gap_analysis": CanonicalSkill(
        key="market_gap_analysis",
        label="market-gap-analysis",
        description="Identify unmet market needs and map them to actionable product bets.",
        category="strategy",
        recommended_roles=("market_researcher", "ceo_planner", "product_manager"),
        priority=1,
    ),
    "architecture_decision_records": CanonicalSkill(
        key="architecture_decision_records",
        label="architecture-decision-records",
        description="Capture design trade-offs and rationale in decision records.",
        category="architecture",
        recommended_roles=("product_manager", "architect", "code_reviewer"),
        priority=2,
    ),
    "backend_patterns": CanonicalSkill(
        key="backend_patterns",
        label="backend-patterns",
        description="Apply robust backend design and implementation patterns.",
        category="implementation",
        recommended_roles=("backend_developer", "code_reviewer"),
        priority=1,
    ),
    "frontend_patterns": CanonicalSkill(
        key="frontend_patterns",
        label="frontend-patterns",
        description="Apply scalable UI architecture patterns and state-management discipline.",
        category="implementation",
        recommended_roles=("ui_ux_designer", "frontend_developer", "architect", "code_reviewer"),
        priority=1,
    ),
    "ux_design": CanonicalSkill(
        key="ux_design",
        label="ux-design",
        description="Design user-centered flows with clear interaction and usability principles.",
        category="design",
        recommended_roles=("ui_ux_designer", "frontend_developer", "product_manager", "qa_engineer"),
        priority=1,
    ),
    "accessibility_review": CanonicalSkill(
        key="accessibility_review",
        label="accessibility-review",
        description="Enforce accessibility requirements and inclusive design checks.",
        category="design",
        recommended_roles=(
            "ui_ux_designer",
            "frontend_developer",
            "qa_engineer",
            "code_reviewer",
        ),
        priority=1,
    ),
    "coding_standards": CanonicalSkill(
        key="coding_standards",
        label="coding-standards",
        description="Enforce consistency in style, naming, structure, and maintainability.",
        category="quality",
        recommended_roles=("backend_developer", "qa_engineer", "code_reviewer"),
        priority=1,
    ),
    "context_budget": CanonicalSkill(
        key="context_budget",
        label="context-budget",
        description="Manage context and output size to preserve relevance and precision.",
        category="communication",
        recommended_roles=(
            "product_manager",
            "architect",
            "ui_ux_designer",
            "backend_developer",
            "security_engineer",
            "qa_engineer",
            "code_reviewer",
            "devops_engineer",
        ),
        priority=1,
    ),
    "deep_research": CanonicalSkill(
        key="deep_research",
        label="deep-research",
        description="Perform evidence-backed research before implementation decisions.",
        category="analysis",
        recommended_roles=("product_manager", "architect", "code_reviewer"),
        priority=2,
    ),
    "deployment_patterns": CanonicalSkill(
        key="deployment_patterns",
        label="deployment-patterns",
        description="Use repeatable production deployment strategies with rollback readiness.",
        category="operations",
        recommended_roles=("devops_engineer", "backend_developer"),
        priority=1,
    ),
    "docker_patterns": CanonicalSkill(
        key="docker_patterns",
        label="docker-patterns",
        description="Containerize services with secure and efficient Docker conventions.",
        category="operations",
        recommended_roles=("devops_engineer", "backend_developer"),
        priority=1,
    ),
    "documentation_lookup": CanonicalSkill(
        key="documentation_lookup",
        label="documentation-lookup",
        description="Verify implementation details against authoritative documentation.",
        category="analysis",
        recommended_roles=("architect", "backend_developer", "qa_engineer", "code_reviewer"),
        priority=2,
    ),
    "e2e_testing": CanonicalSkill(
        key="e2e_testing",
        label="e2e-testing",
        description="Cover critical paths with end-to-end validation.",
        category="testing",
        recommended_roles=("qa_engineer", "backend_developer"),
        priority=1,
    ),
    "git_workflow": CanonicalSkill(
        key="git_workflow",
        label="git-workflow",
        description="Use clear commit sequencing and branch hygiene.",
        category="operations",
        recommended_roles=("backend_developer", "code_reviewer", "devops_engineer"),
        priority=3,
    ),
    "github_ops": CanonicalSkill(
        key="github_ops",
        label="github-ops",
        description="Use PR, CI, and repository workflow best practices.",
        category="operations",
        recommended_roles=("code_reviewer", "devops_engineer", "backend_developer"),
        priority=2,
    ),
    "prompt_optimizer": CanonicalSkill(
        key="prompt_optimizer",
        label="prompt-optimizer",
        description="Improve instruction quality with concise and deterministic prompts.",
        category="communication",
        recommended_roles=("product_manager", "architect", "code_reviewer"),
        priority=3,
    ),
    "python_patterns": CanonicalSkill(
        key="python_patterns",
        label="python-patterns",
        description="Follow reliable, idiomatic Python implementation patterns.",
        category="implementation",
        recommended_roles=("backend_developer", "code_reviewer"),
        priority=1,
    ),
    "python_testing": CanonicalSkill(
        key="python_testing",
        label="python-testing",
        description="Use strong Python testing techniques and structure.",
        category="testing",
        recommended_roles=("qa_engineer", "backend_developer", "code_reviewer"),
        priority=1,
    ),
    "database_migrations": CanonicalSkill(
        key="database_migrations",
        label="database-migrations",
        description="Design safe, reversible schema evolution.",
        category="data",
        recommended_roles=("architect", "backend_developer", "devops_engineer"),
        priority=2,
    ),
    "postgres_patterns": CanonicalSkill(
        key="postgres_patterns",
        label="postgres-patterns",
        description="Use robust PostgreSQL design and query practices.",
        category="data",
        recommended_roles=("architect", "backend_developer", "code_reviewer"),
        priority=2,
    ),
    "security_review": CanonicalSkill(
        key="security_review",
        label="security-review",
        description="Review code and architecture from a threat-first perspective.",
        category="security",
        recommended_roles=(
            "architect",
            "security_engineer",
            "code_reviewer",
            "backend_developer",
            "devops_engineer",
        ),
        priority=1,
    ),
    "security_scan": CanonicalSkill(
        key="security_scan",
        label="security-scan",
        description="Run and interpret security scans with remediation guidance.",
        category="security",
        recommended_roles=(
            "security_engineer",
            "code_reviewer",
            "devops_engineer",
            "backend_developer",
        ),
        priority=1,
    ),
    "repo_scan": CanonicalSkill(
        key="repo_scan",
        label="repo-scan",
        description="Assess repository health, risks, and structural hotspots.",
        category="quality",
        recommended_roles=("code_reviewer", "product_manager", "architect"),
        priority=2,
    ),
    "search_first": CanonicalSkill(
        key="search_first",
        label="search-first",
        description="Search existing code and docs before implementing new logic.",
        category="analysis",
        recommended_roles=(
            "product_manager",
            "architect",
            "ui_ux_designer",
            "backend_developer",
            "security_engineer",
            "qa_engineer",
            "code_reviewer",
            "devops_engineer",
        ),
        priority=1,
    ),
    "tdd_workflow": CanonicalSkill(
        key="tdd_workflow",
        label="tdd-workflow",
        description="Apply test-first implementation with regression safety.",
        category="testing",
        recommended_roles=("backend_developer", "qa_engineer"),
        priority=1,
    ),
    "verification_loop": CanonicalSkill(
        key="verification_loop",
        label="verification-loop",
        description="Continuously verify assumptions and results before handoff.",
        category="quality",
        recommended_roles=("backend_developer", "qa_engineer", "code_reviewer"),
        priority=1,
    ),
    "terminal_ops": CanonicalSkill(
        key="terminal_ops",
        label="terminal-ops",
        description="Use shell operations safely and reproducibly.",
        category="operations",
        recommended_roles=("backend_developer", "devops_engineer"),
        priority=3,
    ),
    "codebase_onboarding": CanonicalSkill(
        key="codebase_onboarding",
        label="codebase-onboarding",
        description="Understand architecture and conventions before modifying code.",
        category="analysis",
        recommended_roles=("backend_developer", "qa_engineer", "code_reviewer"),
        priority=2,
    ),
    "agentic_engineering": CanonicalSkill(
        key="agentic_engineering",
        label="agentic-engineering",
        description="Structure multi-step reasoning with explicit checkpoints and validation.",
        category="process",
        recommended_roles=("product_manager", "architect", "code_reviewer"),
        priority=3,
    ),
    "autonomous_loops": CanonicalSkill(
        key="autonomous_loops",
        label="autonomous-loops",
        description="Run iterative refinement loops with bounded stopping criteria.",
        category="process",
        recommended_roles=("backend_developer", "qa_engineer", "code_reviewer"),
        priority=3,
        conflict_with=("strategic_compact",),
    ),
    "strategic_compact": CanonicalSkill(
        key="strategic_compact",
        label="strategic-compact",
        description="Prefer concise outputs and compact strategic summaries.",
        category="communication",
        recommended_roles=(
            "product_manager",
            "architect",
            "backend_developer",
            "qa_engineer",
            "code_reviewer",
            "devops_engineer",
        ),
        priority=2,
        conflict_with=("autonomous_loops",),
    ),
    "quality_nonconformance": CanonicalSkill(
        key="quality_nonconformance",
        label="quality-nonconformance",
        description="Capture and prioritize quality deviations with remediation tracking.",
        category="quality",
        recommended_roles=("qa_engineer", "code_reviewer", "backend_developer"),
        priority=2,
    ),
}


PACK_PROFILES: dict[str, tuple[str, ...]] = {
    "starter": (
        "search_first",
        "context_budget",
        "strategy_planning",
        "market_gap_analysis",
        "coding_standards",
        "frontend_patterns",
        "ux_design",
        "accessibility_review",
        "python_patterns",
        "api_design",
        "tdd_workflow",
        "python_testing",
        "verification_loop",
        "security_review",
        "deployment_patterns",
        "docker_patterns",
        "documentation_lookup",
    ),
    "advanced": tuple(ECC_CANONICAL_SKILLS.keys()),
}


def _validate_bidirectional_conflicts() -> None:
    for skill in ECC_CANONICAL_SKILLS.values():
        for conflict_key in skill.conflict_with:
            other = ECC_CANONICAL_SKILLS.get(conflict_key)
            if not other:
                raise ValueError(
                    f"Invalid conflict key '{conflict_key}' in skill '{skill.key}'"
                )
            if skill.key not in other.conflict_with:
                raise ValueError(
                    f"Conflict relationship must be bidirectional: "
                    f"'{skill.key}' -> '{conflict_key}'"
                )


_validate_bidirectional_conflicts()


def resolve_ecc_pack_labels(profile: str, role_key: str) -> list[str]:
    """Return ECC skill labels for *role_key* under *profile*."""
    selected = PACK_PROFILES.get(profile, PACK_PROFILES["starter"])
    labels: list[str] = []
    for key in selected:
        skill = ECC_CANONICAL_SKILLS.get(key)
        if not skill:
            continue
        if role_key in skill.recommended_roles:
            labels.append(skill.label)
    return labels


def ecc_priority_map() -> dict[str, int]:
    """Return map of skill label -> numeric priority (lower is more important)."""
    return {skill.label: skill.priority for skill in ECC_CANONICAL_SKILLS.values()}


def ecc_conflicts_map() -> dict[str, set[str]]:
    """Return map of skill label -> conflicting skill labels."""
    by_key = ECC_CANONICAL_SKILLS
    mapping: dict[str, set[str]] = {}
    for skill in by_key.values():
        if not skill.conflict_with:
            continue
        conflicts = {by_key[c].label for c in skill.conflict_with if c in by_key}
        if conflicts:
            mapping[skill.label] = conflicts
    return mapping
