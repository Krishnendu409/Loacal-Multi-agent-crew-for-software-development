"""Persistent memory store for autonomous iterations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.fs import atomic_write_text

logger = logging.getLogger(__name__)

# BUG FIX: cap history size to prevent unbounded disk growth
_MAX_HISTORY_ENTRIES = 500
_MAX_ERROR_ENTRIES = 200
_MAX_BEST_SOLUTIONS = 100


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
        try:
            atomic_write_text(path, json.dumps(items, indent=2, ensure_ascii=False))
        except OSError as exc:
            logger.warning("Could not write memory store %s: %s", path, exc)

    def append_history(self, item: dict[str, Any]) -> None:
        items = self._read_list(self.paths.history)
        items.append(item)
        # BUG FIX: trim to cap
        self._write_list(self.paths.history, items[-_MAX_HISTORY_ENTRIES:])

    def append_error(self, item: dict[str, Any]) -> None:
        items = self._read_list(self.paths.errors)
        items.append(item)
        self._write_list(self.paths.errors, items[-_MAX_ERROR_ENTRIES:])

    def set_best_solution(self, item: dict[str, Any]) -> None:
        items = self._read_list(self.paths.best_solutions)
        items.append(item)
        self._write_list(self.paths.best_solutions, items[-_MAX_BEST_SOLUTIONS:])

    def recent_errors(self, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        items = self._read_list(self.paths.errors)
        return items[-limit:]

    def recent_history(self, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        items = self._read_list(self.paths.history)
        return items[-limit:]
