"""Production-oriented autonomous development crew orchestrator."""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

from src.agents.base_agent import Agent
from src.execution.runner import ExecutionResult, ExecutionRunner
from src.execution.sandbox import Sandbox
from src.memory.store import MemoryStore
from src.project.generator import ProjectGenerator
from src.protocol.messages import AgentMessage, AgentResult
from src.tasks.software_dev_tasks import TASKS, Task
from src.utils import display
from src.utils.fs import atomic_write_text, next_versioned_path

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
    "Backend Developer": "backend_developer",
    "Frontend Developer": "frontend_developer",
    "Data/Analytics Engineer": "data_analytics_engineer",
    "QA Engineer": "qa_engineer",
    "Code Reviewer": "code_reviewer",
    "Security Engineer": "security_engineer",
    "Performance Engineer": "performance_engineer",
    "Technical Writer": "technical_writer",
    "SRE / Reliability Engineer": "sre_reliability_engineer",
    "Release Manager": "release_manager",
    "DevOps Engineer": "devops_engineer",
}


class DevCrew:
    # Major issues are weighted higher to prioritize blocking defects first.
    MAJOR_ISSUE_WEIGHT = 10
    ISSUE_WEIGHT = 1

    def __init__(
        self,
        agents: list[Agent],
        output_dir: str | Path = "output",
        save_individual: bool = True,
        save_report: bool = True,
        max_fix_iterations: int = 3,
        stop_on_no_major_issues: bool = True,
        blocking_severities: tuple[str, ...] = ("critical", "major"),
    ) -> None:
        self.agents = agents
        self.output_dir = Path(output_dir)
        self.save_individual = save_individual
        self.save_report = save_report
        self.max_fix_iterations = max(0, max_fix_iterations)
        self.stop_on_no_major_issues = stop_on_no_major_issues
        self.blocking_severities = {s.lower() for s in blocking_severities}
        self._run_manifest: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def kickoff(
        self,
        requirements: str,
        project_name: str = "project",
        *,
        start_from_role: str | None = None,
        resume_outputs: dict[str, str] | None = None,
        research_context: str = "",
    ) -> dict[str, str]:
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
            start_from_role=start_from_role,
            resume_outputs=resume_outputs,
            research_context=research_context,
        )

    def kickoff_with_strategy_gate(
        self,
        requirements: str,
        project_name: str = "project",
        *,
        require_strategy_approval: bool = True,
        strategy_approval_callback: Callable[[dict[str, str]], bool] | None = None,
        start_from_role: str | None = None,
        resume_outputs: dict[str, str] | None = None,
        research_context: str = "",
    ) -> dict[str, str]:
        return self._kickoff_internal(
            requirements=requirements,
            project_name=project_name,
            require_strategy_approval=require_strategy_approval,
            strategy_approval_callback=strategy_approval_callback,
            start_from_role=start_from_role,
            resume_outputs=resume_outputs,
            research_context=research_context,
        )

    def _kickoff_internal(
        self,
        *,
        requirements: str,
        project_name: str,
        require_strategy_approval: bool,
        strategy_approval_callback: Callable[[dict[str, str]], bool] | None,
        start_from_role: str | None,
        resume_outputs: dict[str, str] | None,
        research_context: str,
    ) -> dict[str, str]:
        outputs: dict[str, str] = {}
        context_parts: list[str] = []
        completed_roles: set[str] = set()
        role_to_key = {role: key for role, key in _ROLE_TO_TASK_KEY.items()}
        start_order = -1
        if start_from_role:
            start_key = role_to_key.get(start_from_role)
            if start_key is None:
                raise ValueError(
                    f"Unknown start_from_role '{start_from_role}'. "
                    f"Known roles: {sorted(_ROLE_TO_TASK_KEY.keys())}"
                )
            start_order = AGENT_ORDER.index(start_key)

        self._initialize_manifest(
            project_name=project_name,
            start_from_role=start_from_role,
            requirements=requirements,
        )
        if resume_outputs:
            self._seed_resume_context(resume_outputs, outputs, context_parts, completed_roles)

        role_to_agent = {agent.role: agent for agent in self.agents}
        role_messages: dict[str, AgentMessage] = {}
        outputs: dict[str, str] = {}

        strategy_roles = [
            "CEO Planner",
            "Market Researcher",
            "Product Manager",
            "Software Architect",
        ]

        for role in strategy_roles:
            agent = role_to_agent.get(role)
            if agent is None:
                continue
            if not self._should_run_role(role, start_order, role_to_key):
                completed_roles.add(role)
                continue
            self._execute_agent(
                agent=agent,
                requirements=self._requirements_with_research(requirements, research_context),
                context_parts=context_parts,
                outputs=outputs,
                project_name=project_name,
            )
            outputs[role] = result.raw_text
            memory.append_history(
                {
                    "phase": "strategy",
                    "role": role,
                    "status": result.status,
                    "issues": result.issues,
                }
            )
            self._save_agent_artifacts(project_name, role, result, iteration=1)

        if require_strategy_approval and strategy_approval_callback:
            strategy_outputs = {role: outputs[role] for role in strategy_roles if role in outputs}
            if not strategy_approval_callback(strategy_outputs):
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
            if not self._should_run_role(role, start_order, role_to_key):
                completed_roles.add(role)
                continue
            self._execute_agent(
                agent=agent,
                requirements=self._requirements_with_research(requirements, research_context),
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
                # On the last allowed iteration, skip reviews that would produce
                # findings with no subsequent fix pass – they add noise without
                # any actionable outcome.
                is_last_iteration = iteration >= self.max_fix_iterations
                can_apply_fix = implementation_agents and not is_last_iteration
                if is_last_iteration and not implementation_agents:
                    break

                round_findings: list[str] = []
                for review_agent in [performance_agent, qa_agent, security_agent, reviewer_agent]:
                    if review_agent is None:
                        continue
                    if not self._should_run_role(review_agent.role, start_order, role_to_key):
                        completed_roles.add(review_agent.role)
                        continue
                    response = self._execute_agent(
                        agent=review_agent,
                        requirements=self._requirements_with_research(requirements, research_context),
                        context_parts=context_parts,
                        outputs=outputs,
                        project_name=project_name,
                        must_address=must_address if must_address else None,
                    )

            execution_result = self._execute_project_checks(runner, project_dir)
            memory.append_history(
                {
                    "phase": "execution",
                    "iteration": iteration,
                    "ok": execution_result.ok,
                    "returncode": execution_result.returncode,
                }
            )
            if not execution_result.ok:
                memory.append_error(
                    {
                        "iteration": iteration,
                        "stderr": execution_result.stderr[-5000:],
                        "stdout": execution_result.stdout[-5000:],
                    }
                )

            review_findings: list[str] = []
            for role in review_roles:
                agent = role_to_agent.get(role)
                if agent is None:
                    continue
                task = self._task_for(agent)
                review_input = self._compose_input_payload(
                    role_messages,
                    requirements,
                    must_address,
                    execution_result=execution_result,
                )
                result = self._execute_structured_agent(
                    agent=agent,
                    requirements=requirements,
                    incoming=review_input,
                    task_description=task.render(requirements=requirements),
                )
                role_messages[role] = self._to_message(task=task.title, input_payload=review_input, result=result)
                outputs[role] = result.raw_text
                self._save_agent_artifacts(project_name, role, result, iteration=iteration)
                review_findings.extend(result.issues)

            combined_issues = list(dict.fromkeys([*review_findings, *self._execution_issues(execution_result)]))
            major_issues = [i for i in combined_issues if self._is_major(i)]
            score = (len(major_issues) * self.MAJOR_ISSUE_WEIGHT) + (
                len(combined_issues) * self.ISSUE_WEIGHT
            )

            if score < best_score:
                best_score = score
                best_snapshot = {
                    "iteration": iteration,
                    "issues": combined_issues,
                    "execution_ok": execution_result.ok,
                    "outputs": outputs.copy(),
                }

            memory.set_best_solution(
                {
                    "iteration": iteration,
                    "score": score,
                    "major_issue_count": len(major_issues),
                    "execution_ok": execution_result.ok,
                }
            )

                if not can_apply_fix:
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
                        requirements=self._requirements_with_research(requirements, research_context),
                        context_parts=context_parts,
                        outputs=outputs,
                        project_name=project_name,
                        task_description=fix_task,
                        must_address=must_address,
                    )
                    completed_roles.add(implementation_agent.role)

            must_address = combined_issues

        executed_roles = set(outputs.keys())
        for agent in self.agents:
            if agent.role in executed_roles:
                continue
            if previous_role:
                display.print_handoff(previous_role, agent.role)
            if not self._should_run_role(agent.role, start_order, role_to_key):
                previous_role = agent.role
                completed_roles.add(agent.role)
                continue
            self._execute_agent(
                agent=agent,
                requirements=self._requirements_with_research(requirements, research_context),
                context_parts=context_parts,
                outputs=outputs,
                project_name=project_name,
            )
            role_messages[agent.role] = self._to_message(
                task=task.title,
                input_payload=incoming,
                result=result,
            )
            outputs[agent.role] = result.raw_text
            self._save_agent_artifacts(project_name, agent.role, result, iteration=1)

        snapshot_outputs = best_snapshot.get("outputs") if isinstance(best_snapshot, dict) else None
        final_outputs = snapshot_outputs if isinstance(snapshot_outputs, dict) else outputs.copy()
        for role, value in outputs.items():
            final_outputs.setdefault(role, value)
        if self.save_report:
            self._save_final_report(project_name, requirements, outputs)
        self._save_run_manifest(project_name, outputs)

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
        return runner.run(["python", "-m", "pytest", "tests", "-q"], cwd=project_dir)

    @staticmethod
    def _summarize_response(response: str, max_chars: int = 900) -> str:
        """Summarize *response* without destructive mid-string truncation.

        Strategy: keep the first ``head`` characters (which usually contain the
        Summary and Key-Decisions sections) and, when the text is long, append
        the last ``tail`` characters so important handoff notes are preserved.
        A clear ellipsis marker separates the two parts so readers know content
        was omitted.
        """
        cleaned = " ".join(response.split())
        if len(cleaned) <= max_chars:
            return cleaned
        # Reserve space for the separator token " […] "
        separator = " […] "
        head = max_chars * 2 // 3
        tail = max_chars - head - len(separator)
        if tail <= 0:
            return cleaned[:head] + "…"
        return cleaned[:head] + separator + cleaned[-tail:]

    @staticmethod
    def _execution_issues(result: ExecutionResult) -> list[str]:
        if result.ok:
            return []
        issues = [f"[Major] Execution failed with return code {result.returncode}."]
        if result.timed_out:
            issues.append("[Major] Test execution timed out.")
        if result.stderr.strip():
            issues.append(f"[Major] stderr: {result.stderr.strip()[:500]}")
        return issues

    def _has_blocking_issues(self, findings: list[str]) -> bool:
        for item in findings:
            lower = item.lower()
            for severity in self.blocking_severities:
                if f"[{severity}]" in lower:
                    return True
                if lower.startswith(f"{severity}:"):
                    return True
        return False

    def _should_run_role(self, role: str, start_order: int, role_to_key: dict[str, str]) -> bool:
        if start_order < 0:
            return True
        key = role_to_key.get(role)
        if key is None:
            return True
        return AGENT_ORDER.index(key) >= start_order

    @staticmethod
    def _requirements_with_research(requirements: str, research_context: str) -> str:
        if not research_context.strip():
            return requirements
        return f"{requirements}\n\n---\n\n{research_context}"

    def _render_fix_task(self, requirements: str, iteration: int, reviewer_roles: list[str]) -> str:
        review_scope = ", ".join(reviewer_roles) if reviewer_roles else "review agents"
        return (
            f"Fix pass #{iteration}: Address only {review_scope} checklist items for the "
            "current implementation. Keep architecture unchanged. Update code and setup "
            "instructions as needed. Be explicit about what was fixed and what remains.\n\n"
            f"Original requirements:\n---\n{requirements}\n---"
        )

    def _execute_structured_agent(
        self,
        *,
        agent: Agent,
        requirements: str,
        incoming: dict[str, Any],
        task_description: str,
        must_address: list[str] | None = None,
    ) -> str:
        task = self._get_task(agent) if task_description is None else None
        title = task.title if task else "Remediate QA/Reviewer issues"
        display.print_agent_start(agent.role, title)

        context = self._build_context(context_parts)
        if task_description is not None:
            rendered_task = task_description
        else:
            if task is None:
                raise ValueError(f"No task available for role '{agent.role}'")
            rendered_task = task.render(requirements=requirements)
        started = time.perf_counter()
        status = "success"
        error_text = ""
        try:
            response = agent.execute(
                rendered_task,
                context=context,
                requirements=requirements,
                must_address=must_address,
            )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_text = str(exc)
            self._record_manifest_role(
                role=agent.role,
                status=status,
                duration_ms=int((time.perf_counter() - started) * 1000),
                output="",
                error=error_text,
                model=agent.llm_model or agent.llm.model,
                retries=agent.llm_retries if agent.llm_retries is not None else agent.llm.retries,
            )
            raise
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
        safe_response = _sanitize_agent_output(response)
        outputs[agent.role] = safe_response
        context_parts.append(self._format_context_entry(agent.role, safe_response))
        self._record_manifest_role(
            role=agent.role,
            status=status,
            duration_ms=duration_ms,
            output=safe_response,
            error=error_text,
            model=agent.llm_model or agent.llm.model,
            retries=agent.llm_retries if agent.llm_retries is not None else agent.llm.retries,
        )
        display.print_agent_response(agent.role, safe_response)
        if self.save_individual:
            self._save_response(project_name, agent.role, safe_response)
        return safe_response

    @staticmethod
    def _to_message(task: str, input_payload: dict[str, Any], result: AgentResult) -> AgentMessage:
        return AgentMessage(
            task=task,
            input=input_payload,
            output={
                "files": result.files,
                "steps": result.steps,
                "status": result.status,
                "summary": result.summary,
            },
            issues=result.issues,
            next_steps=result.steps,
        )

    def _task_for(self, agent: Agent) -> Task:
        key = _ROLE_TO_TASK_KEY.get(agent.role)
        if key is None or key not in TASKS:
            raise ValueError(f"No task defined for role '{agent.role}'.")
        return TASKS[key]

    def _ensure_output_dir(self, project_name: str) -> Path:
        safe_name = _safe_filename(project_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"{safe_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_run_dir(self, project_name: str) -> Path:
        if not hasattr(self, "_run_dir"):
            self._run_dir = self._ensure_output_dir(project_name)
        return self._run_dir

    def _save_agent_artifacts(self, project_name: str, role: str, result: AgentResult, iteration: int) -> None:
        if not self.save_individual:
            return
        run_dir = self._get_run_dir(project_name)
        filename = f"{_safe_filename(role)}.md"
        path = run_dir / filename
        # If the file already exists (fix-pass overwrite), rename the original
        # to a versioned backup before writing the new version.
        if path.exists():
            backup = _next_versioned_path(path)
            try:
                path.rename(backup)
                logger.debug("Preserved original as %s", backup)
            except OSError as exc:
                logger.warning("Could not back up %s: %s", path, exc)
        _atomic_write(path, content)
        display.print_saved(str(path))

    def _save_final_report(
        self, project_name: str, requirements: str, outputs: dict[str, str]
    ) -> None:
        run_dir = self._get_run_dir(project_name)
        path = run_dir / "FINAL_REPORT.md"
        lines = [
            f"# {project_name} - Autonomous Crew Report",
            "",
            f"*Generated: {datetime.datetime.now().isoformat(timespec='seconds')}*",
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

        _atomic_write(path, "\n".join(lines))
        display.print_saved(str(path))

    def _initialize_manifest(
        self, *, project_name: str, start_from_role: str | None, requirements: str
    ) -> None:
        self._run_manifest = {
            "project_name": project_name,
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "start_from_role": start_from_role,
            "requirements_chars": len(requirements),
            "roles": [],
        }

    def _record_manifest_role(
        self,
        *,
        role: str,
        status: str,
        duration_ms: int,
        output: str,
        error: str,
        model: str,
        retries: int,
    ) -> None:
        if "roles" not in self._run_manifest:
            self._run_manifest["roles"] = []
        roles = self._run_manifest["roles"]
        if isinstance(roles, list):
            roles.append(
                {
                    "role": role,
                    "status": status,
                    "duration_ms": duration_ms,
                    "output_chars": len(output),
                    "model": str(model),
                    "retries": int(retries) if isinstance(retries, int) else 0,
                    "sections": _extract_structured_sections(output),
                    "error": error,
                }
            )

    def _save_run_manifest(self, project_name: str, outputs: dict[str, str]) -> None:
        run_dir = self._get_run_dir(project_name)
        self._run_manifest["completed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._run_manifest["total_roles"] = len(outputs)
        self._run_manifest["status"] = "completed"
        path = run_dir / "RUN_MANIFEST.json"
        _atomic_write(path, json.dumps(self._run_manifest, indent=2, ensure_ascii=False))
        display.print_saved(str(path))

    def _seed_resume_context(
        self,
        resume_outputs: dict[str, str],
        outputs: dict[str, str],
        context_parts: list[str],
        completed_roles: set[str],
    ) -> None:
        for role, content in resume_outputs.items():
            safe_content = _sanitize_agent_output(content)
            outputs[role] = safe_content
            context_parts.append(self._format_context_entry(role, safe_content))
            completed_roles.add(role)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).lower()


_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_SCRIPT_TAG_RE = re.compile(r"<\s*/?\s*script\b[^>]*>", flags=re.IGNORECASE)
_PROMPT_INJECTION_RE = re.compile(
    r"(ignore\s+previous\s+instructions|disregard\s+all\s+above|override\s+system\s+prompt)",
    flags=re.IGNORECASE,
)


def _sanitize_agent_output(content: str) -> str:
    """Sanitize model output before display/save to reduce unsafe rendering vectors."""
    if not isinstance(content, str):
        return ""
    text = _ANSI_ESCAPE_RE.sub("", content)
    text = _SCRIPT_TAG_RE.sub("[redacted-script-tag]", text)
    text = _PROMPT_INJECTION_RE.sub("[redacted-prompt-injection]", text)
    # Keep tabs/newlines, drop other control characters.
    text = "".join(
        ch for ch in text if ch in ("\n", "\r", "\t") or ord(ch) >= 32
    )
    return text


def _extract_structured_sections(content: str) -> dict[str, str]:
    """Extract markdown heading sections into a machine-readable mapping."""
    sections: dict[str, str] = {}
    current = "full_text"
    buffer: list[str] = []
    for line in content.splitlines():
        if line.strip().startswith("#"):
            sections[current] = "\n".join(buffer).strip()
            current = line.strip().lstrip("#").strip().lower().replace(" ", "_")
            buffer = []
            continue
        buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return {k: v for k, v in sections.items() if v}


def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically (temp-file + rename).

    Ensures directories exist.  Raises ``OSError`` on failure rather than
    silently leaving a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


_MAX_BACKUP_VERSIONS = 999


def _next_versioned_path(path: Path) -> Path:
    """Return the next available versioned backup path for *path*.

    Example: ``foo.md`` → ``foo.md.bak1``, ``foo.md.bak2``, …
    """
    for n in range(1, _MAX_BACKUP_VERSIONS + 1):
        candidate = path.with_suffix(f"{path.suffix}.bak{n}")
        if not candidate.exists():
            return candidate
    return path.with_suffix(f"{path.suffix}.bak_overflow")
