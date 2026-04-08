"""Production-oriented autonomous development crew orchestrator."""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)

from src.agents.base_agent import Agent
from src.crew.state_graph import StateGraph
from src.execution.docker_runner import DockerExecutionRunner
from src.models.schemas import ArchitectHandoffSchema, QuorumJudgeSchema
from src.protocol.messages import parse_structured_result
from src.tasks.software_dev_tasks import TASKS, Task
from src.utils import display
from src.utils.memory import CrewMemory

# ---------------------------------------------------------------------------
# Agent execution order (task keys, used for start_from_role logic)
# ---------------------------------------------------------------------------

AGENT_ORDER: list[str] = [
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
    REVIEW_GRAPH_STEP_MULTIPLIER = 4
    MAX_SYSTEM_RUNNER_ERROR_CHARS = 2000
    ARCHITECT_QUORUM_OPTION_A_MODEL = "qwen2.5:7b-instruct"
    ARCHITECT_QUORUM_OPTION_B_MODEL = "deepseek-coder:6.7b"
    ARCHITECT_QUORUM_JUDGE_MODEL = "phi3:mini"

    def __init__(
        self,
        agents: list[Agent],
        output_dir: str | Path = "output",
        save_individual: bool = True,
        save_report: bool = True,
        max_fix_iterations: int = 3,
        stop_on_no_major_issues: bool = True,
        blocking_severities: tuple[str, ...] = ("critical", "major"),
        enable_architect_quorum: bool = False,
        enable_system_runner: bool = False,
        enable_vector_memory: bool = False,
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.agents = agents
        self.output_dir = Path(output_dir)
        self.save_individual = save_individual
        self.save_report = save_report
        self.max_fix_iterations = max(0, max_fix_iterations)
        self.stop_on_no_major_issues = stop_on_no_major_issues
        self.blocking_severities = {s.lower() for s in blocking_severities}
        self.enable_architect_quorum = enable_architect_quorum
        self.enable_system_runner = enable_system_runner
        self._run_manifest: dict[str, Any] = {}
        self._docker_runner = DockerExecutionRunner() if enable_system_runner else None
        shared_llm = agents[0].llm if agents else None
        self._memory = (
            CrewMemory(
                persist_dir=self.output_dir / ".crew_memory",
                ollama_client=shared_llm,
                embedding_model=embedding_model,
            )
            if enable_vector_memory and shared_llm is not None
            else None
        )

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
        # Ensure each kickoff gets a fresh run directory, even when reusing the same DevCrew instance.
        if hasattr(self, "_run_dir"):
            delattr(self, "_run_dir")
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

        # Phase 1: Strategy agents
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
            if role == "Software Architect":
                self._execute_architect_with_validation(
                    agent=agent,
                    requirements=self._requirements_with_research(requirements, research_context),
                    context_parts=context_parts,
                    outputs=outputs,
                    project_name=project_name,
                )
            else:
                self._execute_agent(
                    agent=agent,
                    requirements=self._requirements_with_research(requirements, research_context),
                    context_parts=context_parts,
                    outputs=outputs,
                    project_name=project_name,
                )
            completed_roles.add(role)

        self._run_review_graph(
            role_to_agent=role_to_agent,
            requirements=self._requirements_with_research(requirements, research_context),
            context_parts=context_parts,
            outputs=outputs,
            project_name=project_name,
            start_order=start_order,
            role_to_key=role_to_key,
        )

        executed_roles = set(outputs.keys())
        for agent in self.agents:
            if agent.role in executed_roles:
                continue
            if not self._should_run_role(agent.role, start_order, role_to_key):
                completed_roles.add(agent.role)
                continue
            self._execute_agent(
                agent=agent,
                requirements=self._requirements_with_research(requirements, research_context),
                context_parts=context_parts,
                outputs=outputs,
                project_name=project_name,
            )
            completed_roles.add(agent.role)

        if self.save_report:
            self._save_final_report(project_name, requirements, outputs)
        self._save_run_manifest(project_name, outputs)

        return outputs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        """Execute *agent*, record the result, and return the raw response."""
        key = _ROLE_TO_TASK_KEY.get(agent.role)
        if key is None or key not in TASKS:
            raise ValueError(
                f"No task defined for role '{agent.role}'. "
                f"Known roles: {list(_ROLE_TO_TASK_KEY.keys())}"
            )
        task = TASKS[key]
        if task_description is None:
            task_description = task.render(requirements=requirements)
        display.print_agent_start(agent.role, task.title)
        context = "\n\n".join(context_parts) if context_parts else ""
        if self._memory and self._memory.enabled:
            recalls = self._memory.search(query=f"{agent.role}: {task_description}", limit=3)
            if recalls:
                recalled = "\n".join(
                    f"- ({item.metadata.get('role', 'unknown')}) {item.text}" for item in recalls
                )
                context = (
                    f"{context}\n\n## Retrieved Project Memory (Top 3)\n\n{recalled}".strip()
                    if context
                    else f"## Retrieved Project Memory (Top 3)\n\n{recalled}"
                )
        started = time.perf_counter()
        status = "success"
        error_text = ""
        response = ""
        try:
            response = agent.execute(
                task_description,
                context=context,
                requirements=requirements,
                must_address=must_address,
            )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_text = str(exc)
            raise
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
        safe_response = _sanitize_agent_output(response)
        outputs[agent.role] = safe_response
        context_parts.append(self._format_context_entry(agent.role, safe_response))
        if self._memory and self._memory.enabled:
            self._memory.add_artifact(role=agent.role, task=task.title, content=safe_response)
        self._record_manifest_role(
            role=agent.role,
            status=status,
            duration_ms=duration_ms,
            output=safe_response,
            error=error_text,
            model=str(getattr(agent.llm, "model", "") or ""),
            retries=agent.llm_retries
            if agent.llm_retries is not None
            else int(getattr(agent.llm, "retries", 0) or 0),
        )
        display.print_agent_response(agent.role, safe_response)
        if self.save_individual:
            self._save_response(project_name, agent.role, safe_response)
        return safe_response

    def _execute_architect_with_validation(
        self,
        *,
        agent: Agent,
        requirements: str,
        context_parts: list[str],
        outputs: dict[str, str],
        project_name: str,
    ) -> str:
        max_attempts = max(1, self.max_fix_iterations + 1)
        validation_feedback: list[str] = []
        for attempt in range(1, max_attempts + 1):
            if self.enable_architect_quorum:
                task = self._get_task(agent)
                display.print_agent_start(agent.role, task.title)
                started = time.perf_counter()
                response = self._execute_architect_quorum(
                    agent=agent,
                    requirements=requirements,
                    context_parts=context_parts,
                )
                duration_ms = int((time.perf_counter() - started) * 1000)
                safe_response = _sanitize_agent_output(response)
                outputs[agent.role] = safe_response
                context_parts.append(self._format_context_entry(agent.role, safe_response))
                self._record_manifest_role(
                    role=agent.role,
                    status="success",
                    duration_ms=duration_ms,
                    output=safe_response,
                    error="",
                    model=(
                        "quorum["
                        f"{self.ARCHITECT_QUORUM_OPTION_A_MODEL}|"
                        f"{self.ARCHITECT_QUORUM_OPTION_B_MODEL}"
                        f"]->{self.ARCHITECT_QUORUM_JUDGE_MODEL}"
                    ),
                    retries=agent.llm_retries
                    if agent.llm_retries is not None
                    else int(getattr(agent.llm, "retries", 0) or 0),
                )
                display.print_agent_response(agent.role, safe_response)
                if self.save_individual:
                    self._save_response(project_name, agent.role, safe_response)
            else:
                response = self._execute_agent(
                    agent=agent,
                    requirements=requirements,
                    context_parts=context_parts,
                    outputs=outputs,
                    project_name=project_name,
                    must_address=validation_feedback if validation_feedback else None,
                )
                safe_response = response

            try:
                ArchitectHandoffSchema.model_validate_json(safe_response)
                return safe_response
            except ValidationError as exc:
                validation_feedback = [
                    "Return strict JSON matching the architect schema.",
                    f"Validation error: {exc}",
                ]
                if attempt >= max_attempts:
                    raise ValueError(
                        "Software Architect output failed structured validation after retries."
                    ) from exc
                context_parts.append(
                    "### Validation Feedback\n\nArchitect output failed schema validation. "
                    "Retry with corrected JSON only."
                )
        raise RuntimeError("Architect validation loop terminated unexpectedly.")

    def _execute_architect_quorum(
        self, *, agent: Agent, requirements: str, context_parts: list[str]
    ) -> str:
        context = "\n\n".join(context_parts) if context_parts else ""
        task = self._get_task(agent).render(requirements=requirements)
        user_parts = [f"## Your Task\n\n{task}"]
        if requirements.strip():
            user_parts.append(f"## Original Stakeholder Requirements\n\n{requirements}")
        if context.strip():
            user_parts.append(f"## Context from previous team members\n\n{context}")
        user_parts.append("Return JSON only that matches the provided schema.")
        user_message = "\n\n---\n\n".join(user_parts)
        schema = ArchitectHandoffSchema.model_json_schema()
        system_prompt = agent.system_prompt()

        def _candidate(model_name: str) -> str:
            return agent.llm.chat(
                system_prompt,
                user_message,
                model=model_name,
                options=agent.llm_options,
                format_schema=schema,
                fallback_models=agent.llm_fallback_models,
                retries_override=agent.llm_retries,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(_candidate, self.ARCHITECT_QUORUM_OPTION_A_MODEL)
            future_b = executor.submit(_candidate, self.ARCHITECT_QUORUM_OPTION_B_MODEL)
            option_a = future_a.result()
            option_b = future_b.result()

        judge_prompt = (
            "You are the architecture judge. Compare two architecture JSON options, apply "
            "compliance/privacy constraints from context, and select A/B or merge.\n\n"
            f"Option A:\n{option_a}\n\nOption B:\n{option_b}\n"
        )
        judged = agent.llm.chat(
            "Return JSON only according to the judge schema.",
            judge_prompt,
            model=self.ARCHITECT_QUORUM_JUDGE_MODEL,
            format_schema=QuorumJudgeSchema.model_json_schema(),
            options={"temperature": 0.1, "num_predict": 1536},
            retries_override=agent.llm_retries,
        )
        result = QuorumJudgeSchema.model_validate_json(judged)
        return result.merged_architecture.model_dump_json(indent=2)

    def _run_review_graph(
        self,
        *,
        role_to_agent: dict[str, Agent],
        requirements: str,
        context_parts: list[str],
        outputs: dict[str, str],
        project_name: str,
        start_order: int,
        role_to_key: dict[str, str],
    ) -> None:
        reviewer_roles = [
            "Performance Engineer",
            "QA Engineer",
            "Security Engineer",
            "Code Reviewer",
        ]
        implementation_roles = [
            "Frontend Developer",
            "Backend Developer",
            "Database Engineer",
            "API Integration Engineer",
            "Data/Analytics Engineer",
        ]
        reviewers = [
            role_to_agent[role]
            for role in reviewer_roles
            if role in role_to_agent and self._should_run_role(role, start_order, role_to_key)
        ]
        implementations = {
            role: role_to_agent[role]
            for role in implementation_roles
            if role in role_to_agent and self._should_run_role(role, start_order, role_to_key)
        }
        if not reviewers:
            return

        state: dict[str, Any] = {
            "requirements": requirements,
            "current_code": "\n\n".join(
                outputs.get(role, "") for role in implementation_roles if role in outputs
            ),
            "test_results": "",
            "revision_count": 0,
            "issues": [],
            "major_issues": [],
            "must_address": [],
            "iteration": 0,
            "fix_role": None,
        }

        graph = StateGraph()

        def _review_node(local_state: dict[str, Any]) -> None:
            must_address = local_state.get("must_address") or None
            for reviewer in reviewers:
                self._execute_agent(
                    agent=reviewer,
                    requirements=requirements,
                    context_parts=context_parts,
                    outputs=outputs,
                    project_name=project_name,
                    must_address=must_address,
                )
            issues = self._extract_issues(outputs, reviewers)
            runner_issue = self._run_system_runner(outputs)
            if runner_issue:
                issues.append(runner_issue)
                local_state["test_results"] = runner_issue
            local_state["issues"] = issues
            local_state["major_issues"] = [i for i in issues if self._is_major(i)]
            local_state["fix_role"] = self._select_fix_role(
                local_state["major_issues"], implementations
            )

        def _review_router(local_state: dict[str, Any]) -> str | None:
            if local_state.get("iteration", 0) >= self.max_fix_iterations:
                return None
            major = local_state.get("major_issues", [])
            if self.stop_on_no_major_issues and not major:
                return None
            if not major:
                return None
            if not local_state.get("fix_role"):
                return None
            return "fix"

        def _fix_node(local_state: dict[str, Any]) -> None:
            role = local_state.get("fix_role")
            if not isinstance(role, str) or role not in implementations:
                return
            implementation = implementations[role]
            local_state["must_address"] = list(local_state.get("major_issues", []))
            task = self._render_fix_task(
                requirements=requirements,
                iteration=int(local_state.get("iteration", 0)) + 1,
                reviewer_roles=[r.role for r in reviewers],
            )
            self._execute_agent(
                agent=implementation,
                requirements=requirements,
                context_parts=context_parts,
                outputs=outputs,
                project_name=project_name,
                task_description=task,
                must_address=local_state["must_address"],
            )
            local_state["iteration"] = int(local_state.get("iteration", 0)) + 1
            local_state["revision_count"] = int(local_state.get("revision_count", 0)) + 1
            local_state["current_code"] = outputs.get(role, "")

        def _fix_router(local_state: dict[str, Any]) -> str | None:
            if local_state.get("iteration", 0) > self.max_fix_iterations:
                return None
            return "review"

        graph.add_node("review", _review_node, router=_review_router)
        graph.add_node("fix", _fix_node, router=_fix_router)
        graph.set_start("review")
        graph.run(
            state,
            max_steps=max(
                10, (self.max_fix_iterations + 1) * self.REVIEW_GRAPH_STEP_MULTIPLIER
            ),
        )

    def _select_fix_role(self, issues: list[str], implementations: dict[str, Agent]) -> str | None:
        if not issues:
            return None
        joined = " ".join(issues).lower()
        if (
            "database" in joined or "sql" in joined or "schema" in joined
        ) and "Database Engineer" in implementations:
            return "Database Engineer"
        if (
            "api" in joined or "endpoint" in joined or "integration" in joined
        ) and "API Integration Engineer" in implementations:
            return "API Integration Engineer"
        if (
            "frontend" in joined or "ui" in joined or "ux" in joined
        ) and "Frontend Developer" in implementations:
            return "Frontend Developer"
        if "Data/Analytics Engineer" in implementations and (
            "analytics" in joined or "metric" in joined
        ):
            return "Data/Analytics Engineer"
        if "Backend Developer" in implementations:
            return "Backend Developer"
        return next(iter(implementations.keys()), None)

    def _run_system_runner(self, outputs: dict[str, str]) -> str | None:
        if not self._docker_runner:
            return None
        generated_files = self._collect_generated_files(outputs)
        result = self._docker_runner.run_pytest(generated_files)
        if result.skipped or result.ok:
            return None
        details = result.stderr.strip() or result.stdout.strip()
        if not details:
            clipped = "Unknown execution failure."
        elif len(details) > self.MAX_SYSTEM_RUNNER_ERROR_CHARS:
            clipped = (
                details[: self.MAX_SYSTEM_RUNNER_ERROR_CHARS]
                + "\n...[truncated by system runner]..."
            )
        else:
            clipped = details
        return f"[critical] System Runner pytest failure: {clipped}"

    @staticmethod
    def _collect_generated_files(outputs: dict[str, str]) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []
        for role in ("Backend Developer", "QA Engineer"):
            text = outputs.get(role)
            if not text:
                continue
            parsed = parse_structured_result(text)
            for file_obj in parsed.files:
                path = file_obj.get("path", "")
                content = file_obj.get("content", "")
                if not path or not isinstance(content, str):
                    continue
                files.append({"path": path, "content": content})
        return files

    @staticmethod
    def _format_context_entry(role: str, response: str) -> str:
        summary = DevCrew._summarize_response(response)
        return f"### {role}\n\n{summary}"

    def _save_response(self, project_name: str, role: str, content: str) -> None:
        run_dir = self._get_run_dir(project_name)
        filename = f"{_safe_filename(role)}.md"
        path = run_dir / filename
        if path.exists():
            backup = _next_versioned_path(path)
            try:
                path.rename(backup)
                logger.debug("Preserved original as %s", backup)
            except OSError as exc:
                logger.warning("Could not back up %s: %s", path, exc)
        _atomic_write(path, content)
        display.print_saved(str(path))

    def _is_major(self, issue: str) -> bool:
        lower = issue.lower()
        for severity in self.blocking_severities:
            if f"[{severity}]" in lower or lower.startswith(f"{severity}:"):
                return True
        return False

    def _extract_issues(self, outputs: dict[str, str], agents: Sequence[Agent | None]) -> list[str]:
        """Extract issue lines from reviewer agent outputs."""
        issues: list[str] = []
        for agent in agents:
            if agent is None:
                continue
            response = outputs.get(agent.role, "")
            for line in response.splitlines():
                stripped = line.strip().lstrip("- ").strip()
                if stripped and self._is_major(stripped) and stripped not in issues:
                    issues.append(stripped)
        return issues

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
        try:
            return AGENT_ORDER.index(key) >= start_order
        except ValueError:
            return True

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
    text = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or ord(ch) >= 32)
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
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
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
