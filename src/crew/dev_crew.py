"""Development crew orchestrator.

``DevCrew`` brings agents, tasks, and the display layer together.  It runs
agents in sequence, passing the accumulated context from all previous agents
to the next one, mimicking real team communication.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

from src.agents.base_agent import Agent
from src.tasks.software_dev_tasks import TASKS, Task
from src.utils import display

# Map agent role name → task key in TASKS
_ROLE_TO_TASK_KEY: dict[str, str] = {
    "Product Manager": "product_manager",
    "Software Architect": "architect",
    "Backend Developer": "backend_developer",
    "QA Engineer": "qa_engineer",
    "Code Reviewer": "code_reviewer",
    "DevOps Engineer": "devops_engineer",
}


class DevCrew:
    """Orchestrates a sequential multi-agent development pipeline.

    Args:
        agents: Ordered list of ``Agent`` instances to run.
        output_dir: Directory where outputs are saved.
        save_individual: Whether to save each agent's response as its own file.
        save_report: Whether to save the final compiled report.
    """

    def __init__(
        self,
        agents: list[Agent],
        output_dir: str | Path = "output",
        save_individual: bool = True,
        save_report: bool = True,
    ) -> None:
        self.agents = agents
        self.output_dir = Path(output_dir)
        self.save_individual = save_individual
        self.save_report = save_report

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def kickoff(self, requirements: str, project_name: str = "project") -> dict[str, str]:
        """Run the full pipeline and return a dict of role → response.

        Args:
            requirements: Raw project requirements provided by the user.
            project_name: Short identifier used for output filenames.
        """
        outputs: dict[str, str] = {}
        context_parts: list[str] = []

        for i, agent in enumerate(self.agents):
            task = self._get_task(agent)
            display.print_agent_start(agent.role, task.title)

            # Build accumulated context string from all previous agent outputs
            context = self._build_context(context_parts)

            # Execute the agent
            task_description = task.render(requirements=requirements)
            response = agent.execute(task_description, context=context)

            outputs[agent.role] = response
            context_parts.append(self._format_context_entry(agent.role, response))

            display.print_agent_response(agent.role, response)

            # Show handoff arrow to the next agent
            if i < len(self.agents) - 1:
                display.print_handoff(agent.role, self.agents[i + 1].role)

            # Optionally persist to disk
            if self.save_individual:
                self._save_response(project_name, agent.role, response)

        display.print_final_summary(outputs)

        if self.save_report:
            self._save_final_report(project_name, requirements, outputs)

        return outputs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_task(self, agent: Agent) -> Task:
        """Look up the task definition for *agent*'s role."""
        key = _ROLE_TO_TASK_KEY.get(agent.role)
        if key is None or key not in TASKS:
            raise ValueError(
                f"No task defined for role '{agent.role}'.  "
                f"Known roles: {list(_ROLE_TO_TASK_KEY.keys())}"
            )
        return TASKS[key]

    @staticmethod
    def _build_context(parts: list[str]) -> str:
        return "\n\n".join(parts)

    @staticmethod
    def _format_context_entry(role: str, response: str) -> str:
        return f"### {role}\n\n{response}"

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _ensure_output_dir(self, project_name: str) -> Path:
        safe_name = _safe_filename(project_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"{safe_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_run_dir(self, project_name: str) -> Path:
        """Return (and create) the per-run output directory.

        Cached after first call so all files for a run go into the same folder.
        """
        if not hasattr(self, "_run_dir"):
            self._run_dir = self._ensure_output_dir(project_name)
        return self._run_dir

    def _save_response(self, project_name: str, role: str, content: str) -> None:
        run_dir = self._get_run_dir(project_name)
        filename = f"{_safe_filename(role)}.md"
        path = run_dir / filename
        path.write_text(content, encoding="utf-8")
        display.print_saved(str(path))

    def _save_final_report(
        self, project_name: str, requirements: str, outputs: dict[str, str]
    ) -> None:
        run_dir = self._get_run_dir(project_name)
        path = run_dir / "FINAL_REPORT.md"

        lines = [
            f"# {project_name} – Development Crew Report",
            "",
            f"*Generated: {datetime.datetime.now().isoformat(timespec='seconds')}*",
            "",
            "---",
            "",
            "## Original Requirements",
            "",
            requirements,
            "",
            "---",
            "",
        ]
        for role, content in outputs.items():
            lines += [f"## {role}", "", content, "", "---", ""]

        path.write_text("\n".join(lines), encoding="utf-8")
        display.print_saved(str(path))


def _safe_filename(name: str) -> str:
    """Convert a string into a filesystem-safe filename fragment."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).lower()
