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
    "product_manager": Task(
        title="Analyse requirements and write product specification",
        description=(
            "A new software project has been requested.  Here are the raw requirements "
            "provided by the stakeholder:\n\n"
            "---\n{requirements}\n---\n\n"
            "Your job is to analyse these requirements thoroughly and produce a complete "
            "product specification document.  Make sure every ambiguity is resolved or "
            "explicitly flagged.  The specification will be handed directly to the "
            "Software Architect, so it must be precise and actionable."
        ),
    ),
    "architect": Task(
        title="Design system architecture",
        description=(
            "Based on the product specification produced by the Product Manager, "
            "design a pragmatic system architecture for this project.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Choose the simplest architecture that meets the requirements.  Document "
            "your technology choices with clear reasoning.  The Backend Developer "
            "will implement the system based on your design."
        ),
    ),
    "backend_developer": Task(
        title="Implement the backend / core application code",
        description=(
            "Based on the product specification and the architecture design, "
            "implement the backend or core application code.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Write complete, working, well-commented code.  Include all necessary "
            "files and a clear setup guide.  The QA Engineer will test your code next."
        ),
    ),
    "qa_engineer": Task(
        title="Create test plan and automated tests",
        description=(
            "Review the implementation produced by the Backend Developer and create "
            "a comprehensive test plan.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Write concrete automated tests that verify the most important behaviours. "
            "Highlight any bugs or quality concerns you find in the implementation."
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
            "improvement suggestion."
        ),
    ),
    "devops_engineer": Task(
        title="Create deployment and operational configuration",
        description=(
            "Based on the full project deliverables, create all the configuration "
            "needed to deploy and operate this system reliably.\n\n"
            "Original requirements:\n---\n{requirements}\n---\n\n"
            "Provide a complete Dockerfile (or Docker Compose), a CI/CD pipeline "
            "definition, and a brief runbook."
        ),
    ),
}
