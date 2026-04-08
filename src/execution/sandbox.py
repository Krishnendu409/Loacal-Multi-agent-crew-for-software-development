"""Execution safety guardrails for local subprocess runs."""

from __future__ import annotations

from pathlib import Path


class SandboxError(RuntimeError):
    """Raised when execution constraints are violated."""


class Sandbox:
    """Simple local sandbox policy for command execution."""

    _ALLOWED_COMMANDS = {
        "python",
        "python3",
        "pytest",
        "pip",
        "pip3",
        "npm",
        "node",
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def validate_command(self, command: list[str]) -> None:
        if not command:
            raise SandboxError("Command cannot be empty.")
        executable = Path(command[0]).name
        if executable not in self._ALLOWED_COMMANDS:
            raise SandboxError(f"Command '{executable}' is not allowed by sandbox policy.")

    def validate_working_dir(self, working_dir: Path) -> Path:
        resolved = working_dir.resolve()
        if self.workspace not in [resolved, *resolved.parents]:
            raise SandboxError(
                f"Working directory '{resolved}' must be inside workspace '{self.workspace}'."
            )
        return resolved
