"""Filesystem helpers with atomic and versioned writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically write *content* to *path* using replace semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def next_versioned_path(base_dir: Path, stem: str, suffix: str) -> Path:
    """Return the next available versioned path like ``<stem>_vN<suffix>``."""
    base_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        candidate = base_dir / f"{stem}_v{version}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1
