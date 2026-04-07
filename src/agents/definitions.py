"""Pre-built agent definitions for every role in the development crew.

Import ``build_agents`` to get a list of agents configured against a given
``OllamaClient`` instance and a mapping of which agents are enabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.base_agent import Agent
from src.skills import resolve_agent_skills

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
        if isinstance(model, str) and model.strip() and _model_is_allowed(model.strip(), allowed):
            agent.llm_model = model.strip()

    fallbacks = llm_config.get("fallbacks", {})
    if isinstance(fallbacks, dict):
        role_fallbacks = fallbacks.get(role_key, [])
        if isinstance(role_fallbacks, list):
            agent.llm_fallback_models = [
                m.strip()
                for m in role_fallbacks
                if isinstance(m, str) and m.strip() and _model_is_allowed(m.strip(), allowed)
            ]

    role_options = llm_config.get("role_options", {})
    if isinstance(role_options, dict):
        options = role_options.get(role_key, {})
        if isinstance(options, dict):
            agent.llm_options = options

    return agent


def _apply_skill_config(
    agent: Agent,
    role_key: str,
    skills_config: dict[str, object] | None,
) -> Agent:
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


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

_AGENT_FACTORIES = {
    "ceo_planner": _ceo_planner,
    "market_researcher": _market_researcher,
    "product_manager": _product_manager,
    "architect": _architect,
    "ui_ux_designer": _ui_ux_designer,
    "frontend_developer": _frontend_developer,
    "backend_developer": _backend_developer,
    "security_engineer": _security_engineer,
    "qa_engineer": _qa_engineer,
    "code_reviewer": _code_reviewer,
    "devops_engineer": _devops_engineer,
}

# Ordered list – defines the communication sequence in the crew
AGENT_ORDER = [
    "ceo_planner",
    "market_researcher",
    "product_manager",
    "architect",
    "ui_ux_designer",
    "frontend_developer",
    "backend_developer",
    "security_engineer",
    "qa_engineer",
    "code_reviewer",
    "devops_engineer",
]


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
