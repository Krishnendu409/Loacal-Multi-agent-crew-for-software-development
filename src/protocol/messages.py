"""Structured message protocol for inter-agent communication."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    task: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    files: list[dict[str, str]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    status: str = "failure"
    summary: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw_text", None)
        return payload


_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
# Keep summaries compact to avoid prompt bloat and excessive disk writes.
MAX_SUMMARY_LENGTH = 3000
# Bound JSON extraction to a practical size for local model outputs.
MAX_PARSE_LENGTH = 200000


def parse_structured_result(raw_text: str) -> AgentResult:
    """Parse a structured JSON response from model output with safe fallback."""
    payload = _extract_json_payload(raw_text)
    files = payload.get("files", []) if isinstance(payload, dict) else []
    steps = payload.get("steps", []) if isinstance(payload, dict) else []
    issues = payload.get("issues", []) if isinstance(payload, dict) else []
    status = payload.get("status", "failure") if isinstance(payload, dict) else "failure"
    summary = payload.get("summary", "") if isinstance(payload, dict) else ""

    cleaned_files: list[dict[str, str]] = []
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            content = item.get("content")
            if isinstance(path, str) and path.strip() and isinstance(content, str):
                cleaned_files.append({"path": path.strip(), "content": content})

    return AgentResult(
        files=cleaned_files,
        steps=[s for s in steps if isinstance(s, str)] if isinstance(steps, list) else [],
        issues=[i for i in issues if isinstance(i, str)] if isinstance(issues, list) else [],
        status=status if status in {"success", "failure"} else "failure",
        summary=summary if isinstance(summary, str) else "",
        raw_text=raw_text,
    )


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()

    match = _JSON_BLOCK_RE.search(raw_text)
    candidate = (match.group(1).strip() if match else raw_text)[:MAX_PARSE_LENGTH]

    # Try direct parsing first.
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Attempt to locate first JSON object region.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        fragment = candidate[start : end + 1]
        try:
            parsed = json.loads(fragment)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {
        "files": [],
        "steps": [],
        "issues": ["Model output was not valid structured JSON."],
        "status": "failure",
        "summary": raw_text[:MAX_SUMMARY_LENGTH],
    }


def render_message_block(message: AgentMessage) -> str:
    """Render a protocol message as JSON for prompt context."""
    return json.dumps(message.to_dict(), indent=2, ensure_ascii=False)
