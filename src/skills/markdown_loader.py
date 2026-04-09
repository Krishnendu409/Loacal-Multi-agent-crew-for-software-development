"""Loads per-role skill text from markdown files in src/skills/markdown/."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MARKDOWN_DIR = Path(__file__).parent / "markdown"

_ROLE_KEY_TO_FILE: dict[str, str] = {
    "ceo_planner": "strategy_planning.md",
    "market_researcher": "market_research.md",
    "product_manager": "product_management.md",
    "architect": "system_design.md",
    "ui_ux_designer": "frontend_engineering.md",
    "database_engineer": "backend_engineering.md",
    "api_integration_engineer": "backend_engineering.md",
    "frontend_developer": "frontend_engineering.md",
    "backend_developer": "backend_engineering.md",
    "data_analytics_engineer": "backend_engineering.md",
    "performance_engineer": "debugging.md",
    "security_engineer": "security_review.md",
    "qa_engineer": "testing.md",
    "code_reviewer": "critic_review.md",
    "technical_writer": "backend_engineering.md",
    "sre_reliability_engineer": "backend_engineering.md",
    "release_manager": "strategy_planning.md",
    "devops_engineer": "backend_engineering.md",
    "compliance_privacy_specialist": "security_review.md",
    "customer_support_feedback_analyst": "market_research.md",
}


class SkillMarkdownLoader:
    """Loads a skill description from a markdown file for a given role key."""

    def load_for_role(self, role_key: str) -> list[str]:
        """Return a list containing the markdown skill text for *role_key*, or empty."""
        filename = _ROLE_KEY_TO_FILE.get(role_key)
        if not filename:
            return []
        path = _MARKDOWN_DIR / filename
        if not path.exists():
            logger.debug("Skill markdown file not found: %s", path)
            return []
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                # BUG FIX: truncate to 300 chars max to avoid prompt bloat on small models
                return [text[:300]]
        except OSError as exc:
            logger.debug("Could not read skill markdown %s: %s", path, exc)
        return []
