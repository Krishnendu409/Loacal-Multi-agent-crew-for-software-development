"""Docker-backed isolated test execution for generated artifacts."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DockerRunResult:
    ok: bool
    stdout: str
    stderr: str
    skipped: bool = False


class DockerExecutionRunner:
    def __init__(self, image: str = "python:3.12-slim") -> None:
        self.image = image

    def run_pytest(self, files: list[dict[str, str]]) -> DockerRunResult:
        if not files:
            return DockerRunResult(ok=True, stdout="", stderr="", skipped=True)
        test_files = [
            f for f in files if f.get("path", "").startswith("test") or "/test" in f.get("path", "")
        ]
        if not test_files:
            return DockerRunResult(ok=True, stdout="", stderr="", skipped=True)

        try:
            import docker  # type: ignore[import-untyped]
        except Exception:
            return DockerRunResult(
                ok=True,
                stdout="",
                stderr="Docker SDK not installed; skipped system-runner execution.",
                skipped=True,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for item in files:
                rel_path = item.get("path", "")
                content = item.get("content", "")
                if not rel_path or not isinstance(content, str):
                    continue
                rel = Path(rel_path)
                if rel.is_absolute() or ".." in rel.parts:
                    continue
                target = (root / rel_path).resolve()
                if root not in target.parents and target != root:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            client = docker.from_env()
            try:
                logs = client.containers.run(
                    self.image,
                    command="sh -lc 'pip install -q pytest && pytest -q'",
                    volumes={str(root): {"bind": "/workspace", "mode": "rw"}},
                    working_dir="/workspace",
                    remove=True,
                    stderr=True,
                    stdout=True,
                )
                output = (
                    logs.decode("utf-8", errors="replace") if isinstance(logs, bytes) else str(logs)
                )
                return DockerRunResult(ok=True, stdout=output, stderr="")
            except Exception as exc:
                return DockerRunResult(ok=False, stdout="", stderr=str(exc))
