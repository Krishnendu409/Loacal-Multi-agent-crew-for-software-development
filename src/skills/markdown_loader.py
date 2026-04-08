"""Markdown skill loader for role-scoped reusable skills."""

from __future__ import annotations

from pathlib import Path

_ROLE_SKILL_MAP: dict[str, list[str]] = {
    "ceo_planner": ["strategy_planning.md"],
    "market_researcher": ["market_research.md"],
    "product_manager": ["product_management.md"],
    "architect": ["system_design.md"],
    "backend_developer": ["backend_engineering.md", "debugging.md"],
    "frontend_developer": ["frontend_engineering.md", "debugging.md"],
    "qa_engineer": ["testing.md"],
    "code_reviewer": ["critic_review.md", "debugging.md"],
    "security_engineer": ["security_review.md"],
}


class SkillMarkdownLoader:
    def __init__(self, skills_root: Path | None = None) -> None:
        self.skills_root = skills_root or Path(__file__).resolve().parent / "markdown"

    def load_for_role(self, role_key: str) -> list[str]:
        files = _ROLE_SKILL_MAP.get(role_key, [])
        loaded: list[str] = []
        for filename in files:
            path = self.skills_root / filename
            if not path.exists():
                continue
            loaded.append(path.read_text(encoding="utf-8").strip())
        return loaded

    def referenced_files(self, role_key: str) -> list[str]:
        return list(_ROLE_SKILL_MAP.get(role_key, []))
