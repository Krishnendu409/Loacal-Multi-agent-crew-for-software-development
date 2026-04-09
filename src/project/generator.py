"""Project scaffolding and versioned file generation."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.fs import atomic_write_text, next_versioned_path

logger = logging.getLogger(__name__)


class ProjectGenerator:
    def __init__(self, project_root: Path, versions_root: Path) -> None:
        self.project_root = project_root
        self.versions_root = versions_root
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.versions_root.mkdir(parents=True, exist_ok=True)

    def scaffold(self) -> None:
        dirs = [
            self.project_root / "backend" / "routes",
            self.project_root / "backend" / "models",
            self.project_root / "frontend" / "src",
            self.project_root / "tests",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        defaults = {
            "backend/app.py": ("def health() -> dict[str, str]:\n    return {'status': 'ok'}\n"),
            "tests/test_health.py": (
                "from backend.app import health\n\n\ndef test_health():\n"
                "    assert health() == {'status': 'ok'}\n"
            ),
            "frontend/src/main.js": "console.log('frontend ready');\n",
        }
        for relative_path, content in defaults.items():
            path = self.project_root / relative_path
            if not path.exists():
                atomic_write_text(path, content)

    def write_files(self, files: list[dict[str, str]], version_tag: str) -> list[Path]:
        written: list[Path] = []
        project_root_resolved = self.project_root.resolve()
        for item in files:
            rel = item.get("path", "").strip().lstrip("/")
            content = item.get("content", "")
            if not rel:
                continue
            # BUG FIX: check for path traversal
            if ".." in Path(rel).parts:
                logger.warning("Skipped path-traversal file: %s", rel)
                continue
            target = (project_root_resolved / rel).resolve()
            if not target.is_relative_to(project_root_resolved):
                logger.warning("Skipped out-of-root file: %s", rel)
                continue
            atomic_write_text(target, content)
            written.append(target)

            version_stem = rel.replace("/", "_").replace(".", "_")
            version_path = next_versioned_path(
                self.versions_root, f"{version_stem}_{version_tag}", ".txt"
            )
            atomic_write_text(version_path, content)
        return written
