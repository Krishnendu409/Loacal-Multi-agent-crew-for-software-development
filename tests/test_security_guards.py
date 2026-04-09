"""Security-oriented guardrail tests."""

from __future__ import annotations

import pytest

from src.execution.sandbox import Sandbox, SandboxError
from src.utils.research import fetch_research_context


def test_sandbox_rejects_path_based_executable(tmp_path):
    sandbox = Sandbox(workspace=tmp_path)
    with pytest.raises(SandboxError, match="bare command name"):
        sandbox.validate_command(["/tmp/python", "-m", "pytest"])


def test_sandbox_allows_plain_executable_name(tmp_path):
    sandbox = Sandbox(workspace=tmp_path)
    sandbox.validate_command(["python", "-m", "pytest"])


def test_fetch_research_context_blocks_non_http_schemes():
    context = fetch_research_context(["file:///etc/passwd", "ftp://example.com/data.txt"])
    assert "unsupported URL scheme" in context
    assert "only http/https allowed" in context
