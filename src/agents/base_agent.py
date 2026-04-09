"""Base agent class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from src.models.schemas import StandardAgentHandoffSchema

if TYPE_CHECKING:
    from pydantic import BaseModel
    from src.utils.ollama_client import OllamaClient


_CODE_INSTRUCTION = """\
CRITICAL REQUIREMENT — YOU MUST PRODUCE ACTUAL SOURCE CODE FILES:
Every source code file, configuration file, SQL schema, Dockerfile, test file, \
or script you write MUST be placed as a separate entry in the "files" array \
of your JSON response.
DO NOT write code as plain text inside "handoff_notes" or "summary".
The "files" array is the ONLY place where code belongs.
Each entry must have:
  - "path": the relative file path (e.g. "src/app.py", "tests/test_app.py")
  - "content": the COMPLETE file contents — no stubs, no placeholders, \
no pseudocode.
Example:
  {"path": "src/main.py", "content": "def main():\\n    print('Hello')\\n\\nif __name__ == '__main__':\\n    main()\\n"}
Write complete, runnable files that a developer can use immediately.\
"""

# BUG FIX: Simpler, shorter JSON schema description so phi3:mini can follow it reliably.
_HANDOFF_SCHEMA_INSTRUCTION = """\
Return ONLY valid JSON with exactly these fields (no extra text before or after):
{
  "files": [{"path": "relative/path.ext", "content": "complete file contents"}],
  "steps": ["ordered list of delivery phases"],
  "issues": ["risks or concerns"],
  "status": "success",
  "summary": "One paragraph summary of your output",
  "handoff_notes": "Detailed analysis and recommendations for the next role"
}
Set status to "success" when your response is complete. \
Use "failure" only when you genuinely cannot produce any useful output.\
"""


@dataclass
class Agent:
    """A single member of the development crew."""

    role: str
    goal: str
    backstory: str
    llm: "OllamaClient"
    extra_instructions: str = ""
    skills: list[str] = field(default_factory=list)
    enforce_handoff_sections: bool = True
    produces_code: bool = False
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
        ]
        if self.produces_code:
            parts.append(_CODE_INSTRUCTION)
        parts.append(
            "Communicate professionally and clearly. When writing code, always "
            "use proper fenced code blocks with a language tag, e.g. ```python."
        )
        # BUG FIX: limit skills to avoid exceeding context window on small models
        if self.skills:
            # Only include up to 6 skills in the system prompt to stay within context
            active_skills = self.skills[:6]
            skill_lines = "\n".join(f"- {skill}" for skill in active_skills)
            parts.append(f"Apply these operational skills:\n{skill_lines}")
        if self.enforce_handoff_sections:
            parts.append(_HANDOFF_SCHEMA_INSTRUCTION)
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
        """Run the agent on *task_description*, optionally with prior *context*."""
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
        elif self.enforce_handoff_sections:
            format_schema = StandardAgentHandoffSchema.model_json_schema()
        return self.llm.chat(
            self.system_prompt(),
            user_message,
            model=self.llm_model,
            options=self.llm_options,
            format_schema=format_schema,
            fallback_models=self.llm_fallback_models,
            retries_override=self.llm_retries,
        )
