"""Rich-based terminal display helpers.

Provides a consistent, visually appealing UI for the agent communication log.
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()

# Role → colour mapping so each agent has a distinct look
_ROLE_COLOURS: dict[str, str] = {
    "CEO Planner": "bright_cyan",
    "Market Researcher": "bright_blue",
    "Customer Support/Feedback Analyst": "bright_yellow",
    "Product Manager": "cyan",
    "Compliance & Privacy Specialist": "bright_red",
    "Software Architect": "magenta",
    "UI/UX Designer": "bright_white",
    "Database Engineer": "bright_green",
    "API Integration Engineer": "bright_magenta",
    "Frontend Developer": "bright_magenta",
    "Backend Developer": "green",
    "Data/Analytics Engineer": "bright_cyan",
    "Performance Engineer": "yellow",
    "Security Engineer": "bright_red",
    "QA Engineer": "yellow",
    "Code Reviewer": "blue",
    "Technical Writer": "white",
    "SRE / Reliability Engineer": "bright_black",
    "Release Manager": "bright_blue",
    "DevOps Engineer": "red",
}

_ROLE_EMOJIS: dict[str, str] = {
    "CEO Planner": "🧭",
    "Market Researcher": "📈",
    "Customer Support/Feedback Analyst": "🎧",
    "Product Manager": "📋",
    "Compliance & Privacy Specialist": "⚖️",
    "Software Architect": "🏗️",
    "UI/UX Designer": "🧩",
    "Database Engineer": "🗄️",
    "API Integration Engineer": "🔌",
    "Frontend Developer": "🎨",
    "Backend Developer": "💻",
    "Data/Analytics Engineer": "📊",
    "Performance Engineer": "⚡",
    "Security Engineer": "🛡️",
    "QA Engineer": "🧪",
    "Code Reviewer": "🔍",
    "Technical Writer": "📝",
    "SRE / Reliability Engineer": "🛰️",
    "Release Manager": "📦",
    "DevOps Engineer": "🚀",
}


def _colour_for(role: str) -> str:
    return _ROLE_COLOURS.get(role, "white")


def _emoji_for(role: str) -> str:
    return _ROLE_EMOJIS.get(role, "🤖")


def print_banner() -> None:
    """Print the application banner."""
    console.print()
    console.print(
        Panel(
            Text(
                "🤖  Local Multi-Agent Software Dev Crew  🤖\n"
                "Free • Local • Runs on mid-range laptops",
                justify="center",
                style="bold white",
            ),
            style="bold cyan",
            padding=(1, 4),
        )
    )
    console.print()


def print_project_start(project_name: str, model: str) -> None:
    """Announce a new project run."""
    console.print(
        Panel(
            f"[bold]Project:[/bold] {project_name}\n[bold]Model  :[/bold] {model}",
            title="[bold green]Starting new project[/bold green]",
            style="green",
        )
    )
    console.print()


def print_agent_start(role: str, task_title: str) -> None:
    """Show that an agent has started working."""
    colour = _colour_for(role)
    emoji = _emoji_for(role)
    console.print(
        Rule(
            f"{emoji} [{colour}]{role}[/{colour}] → [italic]{task_title}[/italic]",
            style=colour,
        )
    )


def _render_agent_output(content: str) -> Markdown:
    """Convert a raw agent response to human-readable Markdown.

    When the agent returns a JSON blob that matches the standard handoff
    schema, the individual fields (summary, handoff_notes, steps, issues,
    files) are rendered as formatted Markdown sections so the output is
    easy to read in the terminal.  Plain text / Markdown responses are
    returned unchanged.
    """
    stripped = content.strip()
    if not stripped.startswith("{"):
        return Markdown(content)
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return Markdown(content)
    if not isinstance(data, dict):
        return Markdown(content)

    parts: list[str] = []

    # Status badge – only shown when the model flagged a problem
    status = data.get("status", "success")
    if status == "failure":
        parts.append("⚠️ **Status:** The agent flagged issues — see *Issues & Risks* below.")

    # One-paragraph executive summary
    summary = str(data.get("summary", "")).strip()
    if summary:
        parts.append(f"### Summary\n\n{summary}")

    # Main narrative content (full analysis, recommendations, handoff context)
    handoff = str(data.get("handoff_notes", "")).strip()
    if handoff:
        parts.append(f"### Analysis & Handoff Notes\n\n{handoff}")

    # Ordered execution / delivery steps
    steps = data.get("steps", [])
    if isinstance(steps, list) and steps:
        step_lines = "\n".join(f"- {s}" for s in steps if isinstance(s, str) and s.strip())
        if step_lines:
            parts.append(f"### Steps\n\n{step_lines}")

    # Identified issues, risks, or blockers
    issues = data.get("issues", [])
    if isinstance(issues, list) and issues:
        issue_lines = "\n".join(f"- ⚠️ {i}" for i in issues if isinstance(i, str) and i.strip())
        if issue_lines:
            parts.append(f"### Issues & Risks\n\n{issue_lines}")

    # Generated file paths (content is persisted to disk separately)
    files = data.get("files", [])
    if isinstance(files, list) and files:
        file_lines = "\n".join(
            f"- `{f.get('path', '?')}`" for f in files if isinstance(f, dict) and f.get("path")
        )
        if file_lines:
            parts.append(f"### Generated Files\n\n{file_lines}")

    if not parts:
        # JSON present but no recognisable fields – fall back to raw render
        return Markdown(content)

    return Markdown("\n\n---\n\n".join(parts))


def print_agent_response(role: str, content: str) -> None:
    """Display an agent's response in a styled panel."""
    colour = _colour_for(role)
    emoji = _emoji_for(role)
    console.print(
        Panel(
            _render_agent_output(content),
            title=f"{emoji} [bold {colour}]{role}[/bold {colour}]",
            border_style=colour,
            padding=(1, 2),
        )
    )
    console.print()


def print_handoff(from_role: str, to_role: str) -> None:
    """Show a handoff arrow between two agents."""
    from_colour = _colour_for(from_role)
    to_colour = _colour_for(to_role)
    console.print(
        f"  [{from_colour}]{from_role}[/{from_colour}] "
        f"[dim]──────────────────▶[/dim] "
        f"[{to_colour}]{to_role}[/{to_colour}]"
    )
    console.print()


def print_final_summary(outputs: dict[str, str]) -> None:
    """Print a compact table-like summary of all agent outputs."""
    console.print(
        Rule("[bold green]✅  All agents have completed their work[/bold green]", style="green")
    )
    console.print()
    for role, content in outputs.items():
        colour = _colour_for(role)
        emoji = _emoji_for(role)
        preview = content[:200].replace("\n", " ") + ("…" if len(content) > 200 else "")
        console.print(f"  {emoji} [{colour}]{role}[/{colour}]: [dim]{preview}[/dim]")
    console.print()


def print_saved(path: str) -> None:
    """Confirm a file has been saved."""
    console.print(f"  [dim]💾 Saved → {path}[/dim]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]❌  Error:[/bold red] {message}")
