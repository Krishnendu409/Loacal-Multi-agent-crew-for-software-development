"""Base agent class.

Each agent has a *role*, a *goal*, and a *backstory* that prime the LLM to
behave like a real member of a software development team.  Agents execute
*tasks* and receive the accumulated context from all previous agents so they
can build on each other's work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from src.utils.ollama_client import OllamaClient


@dataclass
class Agent:
    """A single member of the development crew."""

    role: str
    goal: str
    backstory: str
    llm: "OllamaClient"
    # Extra per-agent instructions injected into every system prompt
    extra_instructions: str = ""
    skills: list[str] = field(default_factory=list)
    enforce_handoff_sections: bool = True
    llm_model: str | None = None
    llm_options: dict[str, object] = field(default_factory=dict)
    llm_fallback_models: list[str] = field(default_factory=list)
    llm_retries: int | None = None
    output_schema: type["BaseModel"] | None = None

    def _system_prompt(self) -> str:
        parts = [
            f"You are a {self.role} on a software development team.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            (
                "Communicate professionally and clearly.  When writing code, always "
                "use proper fenced code blocks with a language tag, e.g. ```python."
            ),
        ]
        if self.skills:
            skill_lines = "\n\n".join(f"- Skill:\n{skill}" for skill in self.skills)
            parts.append(f"Apply these operational skills in your response:\n{skill_lines}")
        if self.enforce_handoff_sections:
            parts.append(
                "Return structured, valid JSON only with this schema:\n"
                "{\n"
                '  "files": [{"path": "relative/path.ext", "content": "..." }],\n'
                '  "steps": ["..."],\n'
                '  "issues": ["..."],\n'
                '  "status": "success | failure",\n'
                '  "summary": "...",\n'
                '  "handoff_notes": "Handoff Notes for Next Role"\n'
                "}"
            )
        if self.extra_instructions:
            parts.append(self.extra_instructions)
        return "\n\n".join(parts)

    def system_prompt(self) -> str:
        """Public accessor for the fully composed system prompt."""
        return self._system_prompt()

    def execute(
        self,
        task_description: str,
        context: str = "",
        *,
        requirements: str = "",
        must_address: list[str] | None = None,
    ) -> str:
        """Run the agent on *task_description*, optionally with prior *context*.

        Returns the agent's response as a plain string.
        """
        user_message_parts = [f"## Your Task\n\n{task_description}"]
        if requirements.strip():
            user_message_parts.append(f"## Original Stakeholder Requirements\n\n{requirements}")
        if context.strip():
            user_message_parts.append(f"## Context from previous team members\n\n{context}")
        if must_address:
            checklist = "\n".join(f"- {item}" for item in must_address)
            user_message_parts.append(
                "## Must-Address Checklist from QA/Reviewer\n\n"
                f"{checklist}\n\nAddress every item explicitly in your response."
            )
        user_message_parts.append("Please provide your best professional response now.")
        user_message = "\n\n---\n\n".join(user_message_parts)
        format_schema: dict[str, Any] | None = None
        if self.output_schema is not None:
            format_schema = self.output_schema.model_json_schema()
        return self.llm.chat(
            self.system_prompt(),
            user_message,
            model=self.llm_model,
            options=self.llm_options,
            format_schema=format_schema,
            fallback_models=self.llm_fallback_models,
            retries_override=self.llm_retries,
        )
