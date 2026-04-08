"""Pre-built agent definitions for every role in the development crew.

Import ``build_agents`` to get a list of agents configured against a given
``OllamaClient`` instance and a mapping of which agents are enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from src.agents.base_agent import Agent
from src.skills import SkillMarkdownLoader, resolve_agent_skills

if TYPE_CHECKING:
    from src.utils.ollama_client import OllamaClient


def _allowed_models(llm_config: dict[str, object] | None) -> set[str]:
    if not llm_config:
        return set()
    raw = llm_config.get("allowed_models", [])
    if not isinstance(raw, list):
        return set()
    return {m.strip() for m in raw if isinstance(m, str) and m.strip()}


def _model_is_allowed(model: str, allowed: set[str]) -> bool:
    """Return True when *model* is allowed; empty *allowed* means allow all models."""
    return not allowed or model in allowed


def _apply_llm_role_config(
    agent: Agent,
    role_key: str,
    llm_config: dict[str, object] | None,
) -> Agent:
    if not llm_config:
        return agent

    allowed = _allowed_models(llm_config)

    routing = llm_config.get("routing", {})
    if isinstance(routing, dict):
        model = routing.get(role_key)
        if isinstance(model, str):
            model_clean = model.strip()
            if model_clean and _model_is_allowed(model_clean, allowed):
                agent.llm_model = model_clean

    fallbacks = llm_config.get("fallbacks", {})
    if isinstance(fallbacks, dict):
        role_fallbacks = fallbacks.get(role_key, [])
        if isinstance(role_fallbacks, list):
            cleaned: list[str] = []
            for fallback in role_fallbacks:
                if not isinstance(fallback, str):
                    continue
                fallback_clean = fallback.strip()
                if fallback_clean and _model_is_allowed(fallback_clean, allowed):
                    cleaned.append(fallback_clean)
            agent.llm_fallback_models = cleaned

    role_options = llm_config.get("role_options", {})
    if isinstance(role_options, dict):
        options = role_options.get(role_key, {})
        if isinstance(options, dict):
            agent.llm_options = options

    role_retries = llm_config.get("role_retries", {})
    if isinstance(role_retries, dict):
        retries = role_retries.get(role_key)
        if isinstance(retries, int):
            agent.llm_retries = max(retries, 0)

    return agent


def _apply_skill_config(
    agent: Agent,
    role_key: str,
    skills_config: dict[str, object] | None,
) -> Agent:
    markdown_loader = SkillMarkdownLoader()
    markdown_skills = markdown_loader.load_for_role(role_key)
    if markdown_skills:
        agent.skills = markdown_skills
    else:
        agent.skills = resolve_agent_skills(role_key, skills_config)
    if skills_config and "enforce_handoff_sections" in skills_config:
        agent.enforce_handoff_sections = bool(skills_config.get("enforce_handoff_sections"))
    return agent


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def _ceo_planner(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="CEO Planner",
        goal=(
            "Convert the user's raw problem statement into a strategy-first execution plan "
            "with measurable outcomes and sequencing."
        ),
        backstory=(
            "You are a founder-level strategy leader who aligns product direction, feasibility, "
            "and delivery discipline. You force clarity before execution."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Problem Framing** – clarified statement of the real problem\n"
            "2. **Success Metrics** – concrete, measurable outcomes\n"
            "3. **Execution Plan** – phased plan with priorities\n"
            "4. **Risks & Dependencies** – what can block delivery\n"
            "5. **Decision Gate for User** – explicit options and recommended path\n"
            "6. **Handoff to Market Researcher** – what to validate externally"
        ),
    )


def _market_researcher(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Market Researcher",
        goal=(
            "Identify market gaps, competitor patterns, and positioning opportunities "
            "to sharpen the product strategy."
        ),
        backstory=(
            "You are a product strategy analyst focused on evidence-based market and "
            "competitive insights that directly impact product choices."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Target Users & Jobs-to-be-Done**\n"
            "2. **Competitive Landscape** – strengths/weaknesses\n"
            "3. **Market Gaps** – unmet needs worth targeting\n"
            "4. **Differentiation Strategy** – where we can win\n"
            "5. **Scope Recommendation** – MVP vs next iterations\n"
            "6. **Handoff to Product Manager** – decisions to encode in requirements"
        ),
    )


def _customer_support_feedback_analyst(
    llm: "OllamaClient", llm_config: dict[str, object] | None
) -> Agent:
    return Agent(
        role="Customer Support/Feedback Analyst",
        goal=(
            "Translate user pain points into validated product priorities and explicit "
            "service-quality requirements."
        ),
        backstory=(
            "You are a support insights analyst who converts complaints, feature requests, "
            "and usage friction into actionable product and reliability direction."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Likely User Pain Points** – top issues users will face\n"
            "2. **Supportability Requirements** – diagnostics, logs, error clarity, self-help\n"
            "3. **Feedback Loops** – what telemetry/support channels are needed\n"
            "4. **Priority Recommendations** – what should be fixed first and why\n"
            "5. **Handoff to Product Manager** – support-driven requirements to include"
        ),
    )


def _product_manager(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Product Manager",
        goal=(
            "Translate raw project requirements into a clear, actionable product "
            "specification that the entire development team can follow."
        ),
        backstory=(
            "You are a seasoned Product Manager with 10 years of experience "
            "shipping software products.  You excel at capturing ambiguous ideas, "
            "identifying edge-cases early, and writing structured requirements "
            "documents.  You think about the end-user first, and balance technical "
            "feasibility with business value."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Project Overview** – one-paragraph summary\n"
            "2. **Goals & Non-Goals** – bullet lists\n"
            "3. **Functional Requirements** – numbered list\n"
            "4. **Non-Functional Requirements** – performance, security, scalability\n"
            "5. **User Stories** – 'As a <user>, I want <feature> so that <benefit>'\n"
            "6. **Acceptance Criteria** – how we know the project is done\n"
            "7. **Open Questions** – anything still unclear"
        ),
    )


def _compliance_privacy_specialist(
    llm: "OllamaClient", llm_config: dict[str, object] | None
) -> Agent:
    return Agent(
        role="Compliance & Privacy Specialist",
        goal=(
            "Identify compliance and privacy obligations early and convert them into concrete "
            "engineering and process controls."
        ),
        backstory=(
            "You are a privacy and compliance specialist focused on pragmatic controls, data "
            "minimization, consent handling, and auditability."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Regulatory Scope Assumptions** – likely applicable standards\n"
            "2. **Data Classification & Flows** – what data is sensitive and where it moves\n"
            "3. **Required Controls** – retention, consent, access, deletion, auditability\n"
            "4. **Implementation Constraints** – must-do requirements for architecture/build\n"
            "5. **Handoff to Architect and Security Engineer** – mandatory guardrails"
        ),
    )


def _architect(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Software Architect",
        goal=(
            "Design a pragmatic, maintainable system architecture based on the "
            "product specification."
        ),
        backstory=(
            "You are a Principal Software Architect with deep expertise in "
            "distributed systems, API design, and software patterns.  You choose "
            "the simplest architecture that meets requirements, avoid over-engineering, "
            "and always document the reasoning behind your decisions."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Architecture Overview** – high-level description and diagram (ASCII or textual)\n"
            "2. **Technology Stack** – languages, frameworks, databases, etc.\n"
            "3. **Component Breakdown** – each module/service and its responsibility\n"
            "4. **Data Models** – key entities and relationships\n"
            "5. **API Design** – key endpoints/interfaces\n"
            "6. **Design Decisions & Trade-offs** – why this approach\n"
            "7. **Potential Risks** – scalability, security, or complexity concerns"
        ),
    )


def _ui_ux_designer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="UI/UX Designer",
        goal=(
            "Design user journeys, interface structure, and interaction patterns that maximize "
            "clarity, usability, and accessibility."
        ),
        backstory=(
            "You are a Senior UI/UX Designer who translates product goals into practical user "
            "flows, wireframes, and design decisions that engineering can build reliably."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **User Journey Map** – key personas and flows\n"
            "2. **Information Architecture** – screens and navigation model\n"
            "3. **Interaction & Visual Guidelines** – layout, state changes, feedback behavior\n"
            "4. **Accessibility Requirements** – keyboard, contrast, semantics, error messaging\n"
            "5. **Design Handoff** – implementation-ready guidance for Frontend Developer"
        ),
    )


def _database_engineer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Database Engineer",
        goal=(
            "Design resilient data models, indexing, and migration strategies aligned with "
            "product and architecture requirements."
        ),
        backstory=(
            "You are a database engineer focused on schema quality, integrity, performance, "
            "and safe production data evolution."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Data Model Plan** – entities, keys, constraints\n"
            "2. **Indexing & Query Strategy** – expected access patterns and optimizations\n"
            "3. **Migration Plan** – safe rollout/rollback guidance\n"
            "4. **Data Integrity & Retention Controls** – validation and lifecycle rules\n"
            "5. **Handoff to Backend and Data/Analytics Engineers**"
        ),
    )


def _api_integration_engineer(
    llm: "OllamaClient", llm_config: dict[str, object] | None
) -> Agent:
    return Agent(
        role="API Integration Engineer",
        goal=(
            "Define robust internal/external API integration contracts with retries, failure "
            "handling, and compatibility constraints."
        ),
        backstory=(
            "You are an API integration specialist who prevents brittle integrations through "
            "clear contracts, idempotency strategy, and graceful degradation."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Integration Surface** – upstream/downstream dependencies\n"
            "2. **Contract Definitions** – payloads, validation, versioning expectations\n"
            "3. **Reliability Patterns** – retries, circuit-breaking, idempotency\n"
            "4. **Failure & Recovery Paths** – timeout/error handling decisions\n"
            "5. **Handoff to Frontend, Backend, and SRE Engineers**"
        ),
    )


def _backend_developer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Backend Developer",
        goal=(
            "Implement clean, working backend code that fulfils the architecture "
            "and product requirements."
        ),
        backstory=(
            "You are a Senior Backend Developer who writes idiomatic, well-structured "
            "code with clear comments.  You follow SOLID principles, write defensive "
            "code, handle errors explicitly, and always think about readability and "
            "maintainability."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Implementation Plan** – what you will build and why\n"
            "2. **Code** – complete, runnable source files in fenced code blocks\n"
            "3. **Setup Instructions** – how to install dependencies and run the code\n"
            "4. **Known Limitations** – anything not yet implemented\n"
            "5. **Checklist Coverage** – explicitly map each must-address item to a fix"
        ),
    )


def _data_analytics_engineer(
    llm: "OllamaClient", llm_config: dict[str, object] | None
) -> Agent:
    return Agent(
        role="Data/Analytics Engineer",
        goal=(
            "Design trustworthy analytics instrumentation and reporting foundations that map "
            "to product success metrics."
        ),
        backstory=(
            "You are a data and analytics engineer who designs event schemas, data quality checks, "
            "and metric definitions for reliable decision-making."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Metric Framework** – north-star and supporting metrics\n"
            "2. **Event/Tracking Plan** – events, properties, and ownership\n"
            "3. **Data Quality Controls** – validation and anomaly checks\n"
            "4. **Reporting/Observability Outputs** – dashboards or reporting requirements\n"
            "5. **Handoff to Product Manager and Release Manager**"
        ),
    )


def _performance_engineer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Performance Engineer",
        goal=(
            "Identify latency, throughput, and resource bottlenecks and provide prioritized "
            "performance improvements."
        ),
        backstory=(
            "You are a performance engineer who focuses on measurable bottlenecks and practical "
            "tuning guidance rather than premature optimization."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Performance Budget** – response-time/resource targets\n"
            "2. **Hotspot Analysis** – likely bottlenecks by component\n"
            "3. **Optimization Plan** – prioritized fixes with expected impact\n"
            "4. **Load/Stress Validation Guidance** – how to verify improvements\n"
            "5. **Must-Address Checklist** – [Critical]/[Major]/[Minor] findings"
        ),
    )


def _security_engineer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Security Engineer",
        goal=(
            "Proactively identify and prioritize architecture and implementation security risks, "
            "then provide concrete remediation guidance."
        ),
        backstory=(
            "You are an application security engineer focused on threat modeling, secure defaults, "
            "and practical fixes that teams can apply without overcomplication."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Threat Model Snapshot** – assets, trust boundaries, abuse paths\n"
            "2. **Security Findings** – severity-tagged issues with reasoning\n"
            "3. **Required Fixes** – concrete, prioritized remediation actions\n"
            "4. **Verification Guidance** – how to validate each fix\n"
            "5. **Must-Address Checklist** – bullet list where each item starts with "
            "[Critical], [Major], or [Minor]"
        ),
    )


def _frontend_developer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Frontend Developer",
        goal=(
            "Design and implement a high-quality frontend experience aligned with product "
            "requirements, usability, and accessibility."
        ),
        backstory=(
            "You are a Senior Frontend Engineer who balances UI architecture, performance, "
            "component reusability, accessibility, and user-centered design."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Frontend Plan** – pages/components/state strategy\n"
            "2. **UX & Visual Design Decisions** – layout, interaction, accessibility\n"
            "3. **Code** – complete frontend code in fenced blocks\n"
            "4. **Integration Notes** – API contracts and error states\n"
            "5. **Known Limitations** – what remains for iteration"
        ),
    )


def _qa_engineer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="QA Engineer",
        goal=(
            "Validate that the implementation meets requirements by writing a "
            "comprehensive test plan and concrete test cases."
        ),
        backstory=(
            "You are a QA Engineer who combines manual testing instincts with "
            "automated testing expertise.  You read code critically, find edge-cases "
            "that developers miss, and write tests that actually catch regressions."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Test Strategy** – what types of tests (unit, integration, e2e)\n"
            "2. **Test Cases** – table or numbered list with: ID, description, steps, expected result\n"
            "3. **Automated Tests** – concrete test code in fenced code blocks\n"
            "4. **Edge Cases & Negative Tests** – what should fail gracefully\n"
            "5. **Quality Concerns** – bugs or gaps spotted in the implementation\n"
            "6. **Must-Address Checklist** – bullet list where each item starts with "
            "[Critical], [Major], or [Minor]"
        ),
    )


def _code_reviewer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Code Reviewer",
        goal=(
            "Review all deliverables for code quality, security, performance, "
            "and alignment with requirements, then provide constructive feedback."
        ),
        backstory=(
            "You are a Tech Lead who performs thorough code reviews.  You are "
            "constructive but direct: you praise good patterns and flag issues "
            "with specific, actionable suggestions.  You care about security "
            "vulnerabilities, performance anti-patterns, and long-term maintainability."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Overall Assessment** – brief verdict (Approved / Needs Changes / Blocked)\n"
            "2. **Strengths** – what was done well\n"
            "3. **Issues** – severity (Critical / Major / Minor), location, description, suggestion\n"
            "4. **Security Review** – any vulnerabilities or concerns\n"
            "5. **Performance Review** – bottlenecks or inefficiencies\n"
            "6. **Final Recommendations** – ordered list of the most important actions\n"
            "7. **Must-Address Checklist** – bullet list where each item starts with "
            "[Critical], [Major], or [Minor]"
        ),
    )


def _devops_engineer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="DevOps Engineer",
        goal=(
            "Create deployment and operational configurations so the project can "
            "run reliably in production."
        ),
        backstory=(
            "You are a DevOps Engineer who bridges development and operations.  "
            "You write Dockerfiles, CI/CD pipelines, and infrastructure-as-code "
            "with security and reproducibility in mind."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Deployment Strategy** – containerisation, cloud, or on-prem\n"
            "2. **Dockerfile / Docker Compose** – complete files in fenced code blocks\n"
            "3. **CI/CD Pipeline** – YAML definition (GitHub Actions, GitLab CI, etc.)\n"
            "4. **Environment Variables & Secrets** – what needs to be configured\n"
            "5. **Monitoring & Logging** – recommended tooling\n"
            "6. **Runbook** – how to deploy, roll back, and debug in production"
        ),
    )


def _technical_writer(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Technical Writer",
        goal=(
            "Produce implementation-aligned technical documentation and operational runbooks "
            "that reduce handoff and support friction."
        ),
        backstory=(
            "You are a technical writer specialized in developer docs, operator runbooks, "
            "and troubleshooting guides."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Documentation Inventory** – what docs are required\n"
            "2. **Developer Setup Docs** – install/run/debug guidance\n"
            "3. **Operational Runbooks** – incidents, rollback, escalation\n"
            "4. **User-Facing Notes** – release and change communication\n"
            "5. **Handoff to Release Manager and Customer Support/Feedback Analyst**"
        ),
    )


def _sre_reliability_engineer(
    llm: "OllamaClient", llm_config: dict[str, object] | None
) -> Agent:
    return Agent(
        role="SRE / Reliability Engineer",
        goal=(
            "Define reliability objectives, observability, and incident-response readiness for "
            "safe production operation."
        ),
        backstory=(
            "You are an SRE focused on SLOs, error budgets, alert quality, and resilience "
            "engineering."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Reliability Objectives** – SLO/SLI and error budget assumptions\n"
            "2. **Failure Modes & Mitigations** – dependency and infrastructure risks\n"
            "3. **Observability Plan** – logs, metrics, traces, alerts\n"
            "4. **Incident Response Readiness** – on-call and escalation guidance\n"
            "5. **Handoff to DevOps and Release Manager**"
        ),
    )


def _release_manager(llm: "OllamaClient", llm_config: dict[str, object] | None) -> Agent:
    return Agent(
        role="Release Manager",
        goal=(
            "Coordinate a production-ready release plan across engineering, operations, compliance, "
            "documentation, and customer communication."
        ),
        backstory=(
            "You are a release manager who drives release gating, risk sign-off, and staged rollout "
            "discipline."
        ),
        llm=llm,
        extra_instructions=(
            "Structure your output as:\n"
            "1. **Release Readiness Checklist** – required approvals and artifacts\n"
            "2. **Rollout Strategy** – stages, canaries, rollback criteria\n"
            "3. **Risk Register** – open risks and mitigations\n"
            "4. **Go/No-Go Decision Input** – what must be true before launch\n"
            "5. **Post-Release Validation Plan** – stability and customer-impact checks"
        ),
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

_AGENT_FACTORIES: dict[str, Callable[["OllamaClient", dict[str, object] | None], Agent]] = {
    "ceo_planner": _ceo_planner,
    "market_researcher": _market_researcher,
    "customer_support_feedback_analyst": _customer_support_feedback_analyst,
    "product_manager": _product_manager,
    "compliance_privacy_specialist": _compliance_privacy_specialist,
    "architect": _architect,
    "ui_ux_designer": _ui_ux_designer,
    "database_engineer": _database_engineer,
    "api_integration_engineer": _api_integration_engineer,
    "frontend_developer": _frontend_developer,
    "backend_developer": _backend_developer,
    "data_analytics_engineer": _data_analytics_engineer,
    "performance_engineer": _performance_engineer,
    "security_engineer": _security_engineer,
    "qa_engineer": _qa_engineer,
    "code_reviewer": _code_reviewer,
    "technical_writer": _technical_writer,
    "sre_reliability_engineer": _sre_reliability_engineer,
    "release_manager": _release_manager,
    "devops_engineer": _devops_engineer,
}

# Ordered list – defines the communication sequence in the crew
AGENT_ORDER = [
    "ceo_planner",
    "market_researcher",
    "customer_support_feedback_analyst",
    "product_manager",
    "compliance_privacy_specialist",
    "architect",
    "ui_ux_designer",
    "database_engineer",
    "api_integration_engineer",
    "frontend_developer",
    "backend_developer",
    "data_analytics_engineer",
    "performance_engineer",
    "security_engineer",
    "qa_engineer",
    "code_reviewer",
    "technical_writer",
    "sre_reliability_engineer",
    "release_manager",
    "devops_engineer",
]


def register_agent_role(
    key: str,
    factory: Callable[["OllamaClient", dict[str, object] | None], Agent],
    *,
    before: str | None = None,
    after: str | None = None,
) -> None:
    """Register or replace an agent role factory with optional order placement."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Agent key must be a non-empty string.")
    if before and after:
        raise ValueError("Only one of 'before' or 'after' may be set.")
    _AGENT_FACTORIES[key] = factory
    if key in AGENT_ORDER:
        AGENT_ORDER.remove(key)
    if before:
        if before not in AGENT_ORDER:
            raise ValueError(f"Cannot insert before unknown role '{before}'")
        AGENT_ORDER.insert(AGENT_ORDER.index(before), key)
        return
    if after:
        if after not in AGENT_ORDER:
            raise ValueError(f"Cannot insert after unknown role '{after}'")
        AGENT_ORDER.insert(AGENT_ORDER.index(after) + 1, key)
        return
    AGENT_ORDER.append(key)


def build_agents(
    llm: "OllamaClient",
    enabled: dict[str, bool] | None = None,
    llm_config: dict[str, object] | None = None,
    skills_config: dict[str, object] | None = None,
) -> list[Agent]:
    """Return an ordered list of *Agent* instances for the enabled roles.

    Args:
        llm: Configured ``OllamaClient`` used by every agent.
        enabled: Dict mapping agent key → bool (from config).  Defaults to
            all agents enabled except ``devops_engineer``.
    """
    if enabled is None:
        enabled = {k: True for k in AGENT_ORDER}
        enabled["devops_engineer"] = False

    agents: list[Agent] = []
    for key in AGENT_ORDER:
        if enabled.get(key, False):
            factory = _AGENT_FACTORIES[key]
            agent = factory(llm, llm_config)
            agent = _apply_llm_role_config(agent, key, llm_config)
            agent = _apply_skill_config(agent, key, skills_config)
            agents.append(agent)
    return agents
