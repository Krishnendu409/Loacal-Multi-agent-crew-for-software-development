"""Safe subprocess execution with timeout and captured output."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.execution.sandbox import Sandbox


@dataclass
class ExecutionResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class ExecutionRunner:
    def __init__(self, sandbox: Sandbox, timeout_seconds: int = 120) -> None:
        self.sandbox = sandbox
        self.timeout_seconds = timeout_seconds

    def run(self, command: list[str], cwd: Path) -> ExecutionResult:
        self.sandbox.validate_command(command)
        safe_cwd = self.sandbox.validate_working_dir(cwd)

        try:
            completed = subprocess.run(
                command,
                cwd=str(safe_cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return ExecutionResult(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                command=command,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nExecution timed out.",
                timed_out=True,
            )
