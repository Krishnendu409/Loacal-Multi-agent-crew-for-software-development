"""Production-oriented autonomous development crew orchestrator."""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
    def __init__(
        self,
        agents: list[Agent],
        output_dir: str | Path = "output",
        save_individual: bool = True,
        save_report: bool = True,
        max_fix_iterations: int = 2,
        stop_on_no_major_issues: bool = True,
    ) -> None:
        self.agents = agents
        self.output_dir = Path(output_dir)
        self.save_individual = save_individual
        self.save_report = save_report
        self.max_fix_iterations = max(1, max_fix_iterations)
        self.stop_on_no_major_issues = stop_on_no_major_issues

    def kickoff(self, requirements: str, project_name: str = "project") -> dict[str, str]:
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
        run_dir = self._get_run_dir(project_name)
        project_dir = run_dir / "project"
        versions_dir = run_dir / "outputs"
        memory_dir = run_dir / "memory"

        generator = ProjectGenerator(project_root=project_dir, versions_root=versions_dir)
        generator.scaffold()
        memory = MemoryStore(memory_dir)
        runner = ExecutionRunner(Sandbox(project_dir), timeout_seconds=120)

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
            result = self._execute_structured_agent(
                agent=agent,
                requirements=requirements,
                incoming=self._compose_input_payload(role_messages, requirements),
                task_description=self._task_for(agent).render(requirements=requirements),
            )
            role_messages[role] = self._to_message(
                task=self._task_for(agent).title,
                input_payload=self._compose_input_payload(role_messages, requirements),
                result=result,
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

        implementation_roles = ["Backend Developer", "Frontend Developer"]
        review_roles = ["QA Engineer", "Security Engineer", "Code Reviewer", "Performance Engineer"]

        best_score = float("inf")
        best_snapshot: dict[str, Any] | None = None
        must_address: list[str] = []

        for iteration in range(1, self.max_fix_iterations + 1):
            for role in implementation_roles:
                agent = role_to_agent.get(role)
                if agent is None:
                    continue
                task = self._task_for(agent)
                result = self._execute_structured_agent(
                    agent=agent,
                    requirements=requirements,
                    incoming=self._compose_input_payload(role_messages, requirements, must_address),
                    task_description=task.render(requirements=requirements),
                    must_address=must_address or None,
                )
                role_messages[role] = self._to_message(
                    task=task.title,
                    input_payload=self._compose_input_payload(role_messages, requirements, must_address),
                    result=result,
                )
                outputs[role] = result.raw_text
                self._save_agent_artifacts(project_name, role, result, iteration=iteration)
                if result.files:
                    generator.write_files(result.files, version_tag=f"iter{iteration}_{_safe_filename(role)}")

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
            score = len(major_issues) * 10 + len(combined_issues)

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

            if execution_result.ok and not major_issues:
                break

            if self.stop_on_no_major_issues and not major_issues:
                break

            must_address = combined_issues

        executed_roles = set(outputs.keys())
        for agent in self.agents:
            if agent.role in executed_roles:
                continue
            task = self._task_for(agent)
            incoming = self._compose_input_payload(role_messages, requirements, must_address)
            result = self._execute_structured_agent(
                agent=agent,
                requirements=requirements,
                incoming=incoming,
                task_description=task.render(requirements=requirements),
                must_address=must_address or None,
            )
            role_messages[agent.role] = self._to_message(
                task=task.title,
                input_payload=incoming,
                result=result,
            )
            outputs[agent.role] = result.raw_text
            self._save_agent_artifacts(project_name, agent.role, result, iteration=1)

        final_outputs = best_snapshot["outputs"] if best_snapshot else outputs.copy()
        for role, value in outputs.items():
            final_outputs.setdefault(role, value)
        if self.save_report:
            self._save_final_report(project_name, requirements, final_outputs)
        display.print_final_summary(final_outputs)
        return final_outputs

    def _execute_project_checks(self, runner: ExecutionRunner, project_dir: Path) -> ExecutionResult:
        tests_dir = project_dir / "tests"
        if not tests_dir.exists():
            return ExecutionResult(command=["python", "-m", "pytest", "tests", "-q"], returncode=0, stdout="", stderr="", timed_out=False)
        return runner.run(["python", "-m", "pytest", "tests", "-q"], cwd=project_dir)

    def _compose_input_payload(
        self,
        role_messages: dict[str, AgentMessage],
        requirements: str,
        must_address: list[str] | None = None,
        execution_result: ExecutionResult | None = None,
    ) -> dict[str, Any]:
        compact_upstream: dict[str, Any] = {}
        for role, msg in role_messages.items():
            compact_upstream[role] = {
                "task": msg.task,
                "output": {
                    "status": msg.output.get("status"),
                    "summary": str(msg.output.get("summary", ""))[:1200],
                    "file_count": len(msg.output.get("files", []))
                    if isinstance(msg.output.get("files"), list)
                    else 0,
                },
                "issues": msg.issues[:10],
                "next_steps": msg.next_steps[:10],
            }
        payload: dict[str, Any] = {
            "requirements": requirements,
            "upstream": compact_upstream,
            "must_address": must_address or [],
            "recent_errors": [entry for entry in (must_address or [])[:20]],
        }
        if execution_result is not None:
            payload["execution"] = {
                "command": execution_result.command,
                "returncode": execution_result.returncode,
                "stdout_tail": execution_result.stdout[-3000:],
                "stderr_tail": execution_result.stderr[-3000:],
                "timed_out": execution_result.timed_out,
                "ok": execution_result.ok,
            }
        return payload

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

    @staticmethod
    def _is_major(issue: str) -> bool:
        lower = issue.lower()
        return "[major]" in lower or "[critical]" in lower or lower.startswith("major")

    def _execute_structured_agent(
        self,
        *,
        agent: Agent,
        requirements: str,
        incoming: dict[str, Any],
        task_description: str,
        must_address: list[str] | None = None,
    ) -> AgentResult:
        display.print_agent_start(agent.role, task_description.split("\n")[0][:80])
        message = AgentMessage(task=task_description, input=incoming)
        result = agent.execute_structured(
            task_description=task_description,
            message=message,
            requirements=requirements,
            must_address=must_address,
        )
        display.print_agent_response(agent.role, result.raw_text)
        return result

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
        stem = f"{_safe_filename(role)}_iter{iteration}"
        json_path = next_versioned_path(run_dir, stem, ".json")
        md_path = next_versioned_path(run_dir, stem, ".md")
        atomic_write_text(json_path, json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        atomic_write_text(md_path, result.raw_text)
        display.print_saved(str(json_path))
        display.print_saved(str(md_path))

    def _save_final_report(self, project_name: str, requirements: str, outputs: dict[str, str]) -> None:
        run_dir = self._get_run_dir(project_name)
        path = run_dir / "FINAL_REPORT.md"
        lines = [
            f"# {project_name} – Autonomous Crew Report",
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
        atomic_write_text(path, "\n".join(lines))
        display.print_saved(str(path))


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).lower()
