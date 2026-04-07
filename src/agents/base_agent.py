"""Base agent class.

Each agent has a *role*, a *goal*, and a *backstory* that prime the LLM to
behave like a real member of a software development team.  Agents execute
*tasks* and receive the accumulated context from all previous agents so they
can build on each other's work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        if self.extra_instructions:
            parts.append(self.extra_instructions)
        return "\n\n".join(parts)

    def execute(self, task_description: str, context: str = "") -> str:
        """Run the agent on *task_description*, optionally with prior *context*.

        Returns the agent's response as a plain string.
        """
        user_message_parts = [f"## Your Task\n\n{task_description}"]
        if context.strip():
            user_message_parts.append(
                f"## Context from previous team members\n\n{context}"
            )
        user_message_parts.append(
            "Please provide your best professional response now."
        )
        user_message = "\n\n---\n\n".join(user_message_parts)
        return self.llm.chat(self._system_prompt(), user_message)
