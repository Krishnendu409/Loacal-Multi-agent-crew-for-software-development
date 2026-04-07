"""Task definitions for the software development pipeline.

Each ``Task`` object pairs a human-readable *title* with a *description*
template that an ``Agent`` will execute.  Descriptions may reference the
project requirements string via ``{requirements}``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    """A unit of work assigned to an agent."""

    title: str
    description: str

    def render(self, **kwargs: str) -> str:
        """Interpolate template variables in *description* and return the result."""
        return self.description.format(**kwargs)


# ---------------------------------------------------------------------------
# Task catalogue – one task per agent role
# ---------------------------------------------------------------------------

TASKS: dict[str, Task] = {
    "ceo_planner": Task(
        title="Create strategic execution plan from the user's problem statement",
        description=(
            "You are the first planning authority. Convert the stakeholder problem statement "
            "into a concrete strategy and delivery sequence.\n\n"
            "Raw stakeholder problem statement:\n---\n{requirements}\n---\n\n"
            "Define clear goals, trade-offs, risks, and explicit decision options for the user. "
            "Your output must enable a go/no-go decision before implementation starts. "
            "End with an explicit handoff note for the Market Researcher."
        ),
    ),
    "market_researcher": Task(
        title="Research market gaps and competitor positioning",
        description=(
            "Using the stakeholder requirements and strategy draft, produce market-oriented "
            "analysis that improves product direction.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Identify user segments, competing approaches, and market gaps. Recommend how this "
            "project should differentiate in practical terms. End with an explicit handoff note "
            "for the Product Manager."
        ),
    ),
    "customer_support_feedback_analyst": Task(
        title="Convert support signals into product and reliability requirements",
        description=(
            "Using the stakeholder problem statement and planning context, infer likely customer "
            "support pain points and translate them into implementation requirements.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Prioritize friction points, support workflows, diagnostic visibility, and feedback "
            "loops needed after launch. End with an explicit handoff note for the Product Manager."
        ),
    ),
    "product_manager": Task(
        title="Analyse requirements and write product specification",
        description=(
            "A new software project has been requested.  Here are the raw requirements "
            "provided by the stakeholder:\n\n"
            "---\n{requirements}\n---\n\n"
            "Your job is to analyse these requirements thoroughly and produce a complete "
            "product specification document.  Make sure every ambiguity is resolved or "
            "explicitly flagged.  The specification will be handed directly to the "
            "Software Architect, so it must be precise and actionable.  Keep sections "
            "structured and concise so later agents can consume a compact summary. Include a "
            "final strategy package suitable for user approval before build work starts. End with "
            "an explicit handoff note for the Compliance & Privacy Specialist and Software Architect."
        ),
    ),
    "compliance_privacy_specialist": Task(
        title="Define compliance and privacy guardrails before architecture and build",
        description=(
            "Review planning outputs and define practical compliance/privacy constraints that the "
            "architecture and implementation must satisfy.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Provide data classification assumptions, mandatory controls, and constraints for "
            "retention, consent, access, and auditability. End with explicit handoff notes for "
            "Software Architect and Security Engineer."
        ),
    ),
    "architect": Task(
        title="Design system architecture",
        description=(
            "Based on the product specification produced by the Product Manager, "
            "design a pragmatic system architecture for this project.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Choose the simplest architecture that meets the requirements.  Document "
            "your technology choices with clear reasoning.  UI/UX and engineering roles "
            "will implement the system based on your design.  Include clear trade-offs "
            "and end with explicit handoff notes for the UI/UX Designer and developers."
        ),
    ),
    "ui_ux_designer": Task(
        title="Design user experience and interface system",
        description=(
            "Based on strategy, product requirements, and architecture, design the complete "
            "user experience blueprint before implementation.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Define user journeys, navigation, interaction states, visual consistency rules, "
            "and accessibility requirements. Keep the output implementation-ready and end "
            "with an explicit handoff note for the Frontend Developer."
        ),
    ),
    "database_engineer": Task(
        title="Design database architecture, schema strategy, and migration plan",
        description=(
            "Based on product requirements and architecture, design the persistent data layer.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Define schema entities, indexing strategy, migration approach, and data integrity "
            "rules. Keep output implementation-ready and end with explicit handoff notes for "
            "Backend Developer and Data/Analytics Engineer."
        ),
    ),
    "api_integration_engineer": Task(
        title="Define robust API integration contracts and failure-handling strategy",
        description=(
            "Using the architecture and implementation context, define integration-safe API "
            "contracts and operational handling patterns.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Cover payload contracts, versioning, retries, idempotency, timeout handling, and "
            "degradation strategies. End with explicit handoff notes for Frontend, Backend, and "
            "SRE / Reliability Engineer."
        ),
    ),
    "backend_developer": Task(
        title="Implement the backend / core application code",
        description=(
            "Based on the product specification and the architecture design, "
            "implement the backend or core application code.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Write complete, working, well-commented code.  Include all necessary "
            "files and a clear setup guide.  The QA Engineer will test your code next. "
            "If a must-address checklist is provided, fix only those items. End with "
            "an explicit handoff note for QA and Code Review."
        ),
    ),
    "frontend_developer": Task(
        title="Design and implement frontend UX/UI",
        description=(
            "Based on the strategy, product specification, and architecture design, "
            "implement the frontend user experience.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Provide production-quality UI structure, component logic, and clear interaction "
            "flows. Ensure responsiveness, accessibility, and error handling paths. "
            "If a must-address checklist is provided, fix only those items. End with "
            "an explicit handoff note for Backend, Security Engineer, QA, and Code Review."
        ),
    ),
    "data_analytics_engineer": Task(
        title="Design analytics instrumentation and KPI measurement layer",
        description=(
            "Use strategy and product context to define how success metrics will be captured "
            "reliably from the implementation.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Define event taxonomy, metric formulas, quality checks, and reporting requirements. "
            "End with explicit handoff notes for Product Manager and Release Manager."
        ),
    ),
    "performance_engineer": Task(
        title="Run performance-focused review and optimization guidance",
        description=(
            "Review architecture and implementation outputs and identify bottlenecks and "
            "performance risks.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Define performance budgets, likely hotspots, and prioritized remediations with "
            "severity tags. End with a must-address checklist and explicit handoff note for "
            "Frontend/Backend/Database/API Integration engineers."
        ),
    ),
    "qa_engineer": Task(
        title="Create test plan and automated tests",
        description=(
            "Review the implementation produced by Frontend/Backend developers and create "
            "a comprehensive test plan.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Write concrete automated tests that verify the most important behaviors. "
            "Highlight any bugs or quality concerns you find in the implementation. "
            "Provide a must-address checklist with severity tags and end with an "
            "explicit handoff note for Frontend/Backend developers."
        ),
    ),
    "security_engineer": Task(
        title="Run threat-oriented security review and remediation guidance",
        description=(
            "Review architecture and implementation outputs from the team and produce a "
            "security-focused assessment.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Identify vulnerabilities, insecure defaults, and abuse paths. For each issue, "
            "provide severity and concrete remediation guidance. End with a must-address "
            "checklist and explicit handoff note for Frontend/Backend developers."
        ),
    ),
    "code_reviewer": Task(
        title="Review all deliverables for quality, security, and correctness",
        description=(
            "You have received all deliverables from the team: the product spec, "
            "architecture, implementation, and test plan.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Perform a thorough review.  Be specific: cite file/line when possible, "
            "explain *why* something is an issue, and always provide a concrete "
            "improvement suggestion.  End with a must-address checklist with severities "
            "and a clear handoff note for the Backend Developer (or DevOps if enabled)."
        ),
    ),
    "devops_engineer": Task(
        title="Create deployment and operational configuration",
        description=(
            "Based on the full project deliverables, create all the configuration "
            "needed to deploy and operate this system reliably.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Provide a complete Dockerfile (or Docker Compose), a CI/CD pipeline "
            "definition, and a brief runbook. End with operational handoff notes."
        ),
    ),
    "technical_writer": Task(
        title="Create technical documentation and operational runbooks",
        description=(
            "Using completed technical outputs, produce concise docs for developers, operators, "
            "and customer-facing stakeholders.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Deliver setup docs, troubleshooting guides, and operational runbooks with clear "
            "ownership. End with explicit handoff notes for Release Manager and Customer Support/Feedback Analyst."
        ),
    ),
    "sre_reliability_engineer": Task(
        title="Define reliability objectives and incident-readiness plan",
        description=(
            "Using implementation and operations context, design reliability guardrails for "
            "production operation.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Define SLO/SLI assumptions, observability standards, alerting strategy, and incident "
            "response requirements. End with explicit handoff notes for DevOps Engineer and Release Manager."
        ),
    ),
    "release_manager": Task(
        title="Finalize go/no-go release plan across product, engineering, and operations",
        description=(
            "Consolidate all outputs into a release readiness decision package.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Produce release gates, rollout strategy, rollback criteria, owner assignments, and "
            "post-release validation steps. End with a clear go/no-go recommendation."
        ),
    ),
}
