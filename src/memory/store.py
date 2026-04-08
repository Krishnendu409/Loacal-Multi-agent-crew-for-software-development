"""Persistent memory store for autonomous iterations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.fs import atomic_write_text


@dataclass
class MemoryPaths:
    history: Path
    errors: Path
    best_solutions: Path


class MemoryStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.paths = MemoryPaths(
            history=self.root_dir / "history.json",
            errors=self.root_dir / "errors.json",
            best_solutions=self.root_dir / "best_solutions.json",
        )
        self._ensure_files()

    def _ensure_files(self) -> None:
        for path in (self.paths.history, self.paths.errors, self.paths.best_solutions):
            if not path.exists():
                atomic_write_text(path, "[]")

    def _read_list(self, path: Path) -> list[dict[str, Any]]:
        try:
            content = path.read_text(encoding="utf-8")
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [i for i in parsed if isinstance(i, dict)]
        except (OSError, json.JSONDecodeError):
            return []
        return []

    def _write_list(self, path: Path, items: list[dict[str, Any]]) -> None:
        atomic_write_text(path, json.dumps(items, indent=2, ensure_ascii=False))

    def append_history(self, item: dict[str, Any]) -> None:
        items = self._read_list(self.paths.history)
        items.append(item)
        self._write_list(self.paths.history, items)

    def append_error(self, item: dict[str, Any]) -> None:
        items = self._read_list(self.paths.errors)
        items.append(item)
        self._write_list(self.paths.errors, items)

    def set_best_solution(self, item: dict[str, Any]) -> None:
        items = self._read_list(self.paths.best_solutions)
        items.append(item)
        self._write_list(self.paths.best_solutions, items)

    def recent_errors(self, limit: int = 5) -> list[dict[str, Any]]:
        items = self._read_list(self.paths.errors)
        return items[-max(0, limit) :]

    def recent_history(self, limit: int = 5) -> list[dict[str, Any]]:
        items = self._read_list(self.paths.history)
        return items[-max(0, limit) :]
