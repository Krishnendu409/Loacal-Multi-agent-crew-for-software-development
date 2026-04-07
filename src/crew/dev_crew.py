"""Development crew orchestrator.

``DevCrew`` brings agents, tasks, and the display layer together.  It runs
agents in sequence, passing the accumulated context from all previous agents
to the next one, mimicking real team communication.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any
from typing import Callable

from src.agents.base_agent import Agent
from src.agents.definitions import AGENT_ORDER
from src.tasks.software_dev_tasks import TASKS, Task
from src.utils import display

# Map agent role name → task key in TASKS
_ROLE_TO_TASK_KEY: dict[str, str] = {
    "CEO Planner": "ceo_planner",
    "Market Researcher": "market_researcher",
    "Customer Support/Feedback Analyst": "customer_support_feedback_analyst",
    "Product Manager": "product_manager",
    "Compliance & Privacy Specialist": "compliance_privacy_specialist",
    "Software Architect": "architect",
    "UI/UX Designer": "ui_ux_designer",
    "Database Engineer": "database_engineer",
    "API Integration Engineer": "api_integration_engineer",
    "Frontend Developer": "frontend_developer",
    "Backend Developer": "backend_developer",
    "Data/Analytics Engineer": "data_analytics_engineer",
    "Performance Engineer": "performance_engineer",
    "Security Engineer": "security_engineer",
    "QA Engineer": "qa_engineer",
    "Code Reviewer": "code_reviewer",
    "Technical Writer": "technical_writer",
    "SRE / Reliability Engineer": "sre_reliability_engineer",
    "Release Manager": "release_manager",
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
        max_fix_iterations: int = 1,
        stop_on_no_major_issues: bool = True,
    ) -> None:
        self.agents = agents
        self.output_dir = Path(output_dir)
        self.save_individual = save_individual
        self.save_report = save_report
        self.max_fix_iterations = max(0, max_fix_iterations)
        self.stop_on_no_major_issues = stop_on_no_major_issues

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def kickoff(self, requirements: str, project_name: str = "project") -> dict[str, str]:
        """Run the full pipeline and return a dict of role → response.

        Args:
            requirements: Raw project requirements provided by the user.
            project_name: Short identifier used for output filenames.
        """
        return self._kickoff_internal(
            requirements=requirements,
            project_name=project_name,
            require_strategy_approval=False,
            strategy_approval_callback=None,
        )

    def kickoff_with_strategy_gate(
        self,
        requirements: str,
        project_name: str = "project",
        *,
        require_strategy_approval: bool = True,
        strategy_approval_callback: Callable[[dict[str, str]], bool] | None = None,
    ) -> dict[str, str]:
        """Run the full pipeline with optional strategy approval gate."""
        return self._kickoff_internal(
            requirements=requirements,
            project_name=project_name,
            require_strategy_approval=require_strategy_approval,
            strategy_approval_callback=strategy_approval_callback,
        )

    def _kickoff_internal(
        self,
        *,
        requirements: str,
        project_name: str,
        require_strategy_approval: bool,
        strategy_approval_callback: Callable[[dict[str, str]], bool] | None,
    ) -> dict[str, str]:
        outputs: dict[str, str] = {}
        context_parts: list[str] = []
        completed_roles: set[str] = set()

        role_to_agent = {agent.role: agent for agent in self.agents}

        # Phase 1: Strategy and market planning
        strategy_roles = [
            "CEO Planner",
            "Market Researcher",
            "Customer Support/Feedback Analyst",
            "Product Manager",
            "Compliance & Privacy Specialist",
        ]
        for role in strategy_roles:
            agent = role_to_agent.get(role)
            if agent is None:
                continue
            self._execute_agent(
                agent=agent,
                requirements=requirements,
                context_parts=context_parts,
                outputs=outputs,
                project_name=project_name,
            )
            completed_roles.add(role)

        if require_strategy_approval and strategy_approval_callback:
            strategy_outputs = {role: outputs[role] for role in strategy_roles if role in outputs}
            approved = strategy_approval_callback(strategy_outputs)
            if not approved:
                return outputs

        # Phase 2: Architecture + design + build execution
        for role in [
            "Software Architect",
            "UI/UX Designer",
            "Database Engineer",
            "API Integration Engineer",
            "Frontend Developer",
            "Backend Developer",
            "Data/Analytics Engineer",
        ]:
            agent = role_to_agent.get(role)
            if agent is None:
                continue
            self._execute_agent(
                agent=agent,
                requirements=requirements,
                context_parts=context_parts,
                outputs=outputs,
                project_name=project_name,
            )
            completed_roles.add(role)

        # Phase 3/4: QA + Security + Reviewer findings, then implementation fix pass (bounded)
        performance_agent = role_to_agent.get("Performance Engineer")
        qa_agent = role_to_agent.get("QA Engineer")
        security_agent = role_to_agent.get("Security Engineer")
        reviewer_agent = role_to_agent.get("Code Reviewer")
        database_agent = role_to_agent.get("Database Engineer")
        api_integration_agent = role_to_agent.get("API Integration Engineer")
        analytics_agent = role_to_agent.get("Data/Analytics Engineer")
        frontend_agent = role_to_agent.get("Frontend Developer")
        backend_agent = role_to_agent.get("Backend Developer")
        implementation_agents = [
            agent
            for agent in [
                frontend_agent,
                backend_agent,
                database_agent,
                api_integration_agent,
                analytics_agent,
            ]
            if agent is not None
        ]
        must_address: list[str] = []

        if performance_agent or qa_agent or reviewer_agent or security_agent:
            for iteration in range(self.max_fix_iterations + 1):
                round_findings: list[str] = []
                for review_agent in [performance_agent, qa_agent, security_agent, reviewer_agent]:
                    if review_agent is None:
                        continue
                    response = self._execute_agent(
                        agent=review_agent,
                        requirements=requirements,
                        context_parts=context_parts,
                        outputs=outputs,
                        project_name=project_name,
                        must_address=must_address if must_address else None,
                    )
                    completed_roles.add(review_agent.role)
                    round_findings.extend(self._extract_must_address(response))

                if not round_findings:
                    break

                must_address = round_findings

                if self.stop_on_no_major_issues and not self._has_blocking_issues(must_address):
                    break

                if not implementation_agents or iteration >= self.max_fix_iterations:
                    break

                fix_task = self._render_fix_task(
                    requirements=requirements,
                    iteration=iteration + 1,
                    reviewer_roles=[
                        agent.role
                        for agent in [performance_agent, security_agent, qa_agent, reviewer_agent]
                        if agent is not None
                    ],
                )
                for implementation_agent in implementation_agents:
                    self._execute_agent(
                        agent=implementation_agent,
                        requirements=requirements,
                        context_parts=context_parts,
                        outputs=outputs,
                        project_name=project_name,
                        task_description=fix_task,
                        must_address=must_address,
                    )
                    completed_roles.add(implementation_agent.role)

        # Run any remaining enabled agents (e.g., DevOps) in standard order
        ordered_agents = sorted(
            self.agents,
            key=lambda a: AGENT_ORDER.index(_ROLE_TO_TASK_KEY.get(a.role, "devops_engineer")),
        )
        previous_role: str | None = None
        for agent in ordered_agents:
            if agent.role in completed_roles:
                previous_role = agent.role
                continue
            if previous_role:
                display.print_handoff(previous_role, agent.role)
            self._execute_agent(
                agent=agent,
                requirements=requirements,
                context_parts=context_parts,
                outputs=outputs,
                project_name=project_name,
            )
            completed_roles.add(agent.role)
            previous_role = agent.role

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
                f"No task defined for role '{agent.role}'. "
                f"Known roles: {list(_ROLE_TO_TASK_KEY.keys())}"
            )
        return TASKS[key]

    @staticmethod
    def _build_context(parts: list[str]) -> str:
        return "\n\n".join(parts)

    @classmethod
    def _format_context_entry(cls, role: str, response: str) -> str:
        summary = cls._summarize_response(response)
        return f"### {role}\n\n{summary}"

    @staticmethod
    def _summarize_response(response: str, max_chars: int = 900) -> str:
        cleaned = " ".join(response.split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1] + "…"

    @staticmethod
    def _extract_must_address(response: str) -> list[str]:
        findings: list[str] = []
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        capture = False
        for line in lines:
            lower = line.lower()
            if "must-address checklist" in lower:
                capture = True
                continue
            if capture and lower.startswith("#"):
                capture = False
            if capture and line.startswith(("-", "*")):
                findings.append(line.lstrip("-* ").strip())
                continue
            if re.search(r"\[(critical|major|minor)\]", lower):
                findings.append(line)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for item in findings:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @staticmethod
    def _has_blocking_issues(findings: list[str]) -> bool:
        for item in findings:
            lower = item.lower()
            if "[critical]" in lower or "[major]" in lower:
                return True
            if lower.startswith("critical:") or lower.startswith("major:"):
                return True
        return False

    def _render_fix_task(self, requirements: str, iteration: int, reviewer_roles: list[str]) -> str:
        review_scope = ", ".join(reviewer_roles) if reviewer_roles else "review agents"
        return (
            f"Fix pass #{iteration}: Address only {review_scope} checklist items for the "
            "current implementation. Keep architecture unchanged. Update code and setup "
            "instructions as needed. Be explicit about what was fixed and what remains.\n\n"
            f"Original requirements:\n---\n{requirements}\n---"
        )

    def _execute_agent(
        self,
        *,
        agent: Agent,
        requirements: str,
        context_parts: list[str],
        outputs: dict[str, str],
        project_name: str,
        task_description: str | None = None,
        must_address: list[str] | None = None,
    ) -> str:
        task = self._get_task(agent) if task_description is None else None
        title = task.title if task else "Remediate QA/Reviewer issues"
        display.print_agent_start(agent.role, title)

        context = self._build_context(context_parts)
        rendered_task = task_description or task.render(requirements=requirements)
        response = agent.execute(
            rendered_task,
            context=context,
            requirements=requirements,
            must_address=must_address,
        )
        outputs[agent.role] = response
        context_parts.append(self._format_context_entry(agent.role, response))
        display.print_agent_response(agent.role, response)
        if self.save_individual:
            self._save_response(project_name, agent.role, response)
        return response

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
