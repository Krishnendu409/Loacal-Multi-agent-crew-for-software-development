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
        executable_raw = str(command[0]).strip()
        if not executable_raw:
            raise SandboxError("Command executable cannot be empty.")
        if Path(executable_raw).name != executable_raw:
            raise SandboxError("Executable must be a bare command name, not a path.")
        executable = executable_raw
        if executable not in self._ALLOWED_COMMANDS:
            raise SandboxError(f"Command '{executable}' is not allowed by sandbox policy.")

    def validate_working_dir(self, working_dir: Path) -> Path:
        resolved = working_dir.resolve()
        if self.workspace not in resolved.parents and resolved != self.workspace:
            raise SandboxError(
                f"Working directory '{resolved}' must be inside workspace '{self.workspace}'."
            )
        return resolved
