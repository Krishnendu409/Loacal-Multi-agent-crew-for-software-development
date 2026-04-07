"""Rich-based terminal display helpers.

Provides a consistent, visually appealing UI for the agent communication log.
"""

from __future__ import annotations

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
    "Product Manager": "cyan",
    "Software Architect": "magenta",
    "Frontend Developer": "bright_magenta",
    "Backend Developer": "green",
    "QA Engineer": "yellow",
    "Code Reviewer": "blue",
    "DevOps Engineer": "red",
}

_ROLE_EMOJIS: dict[str, str] = {
    "CEO Planner": "🧭",
    "Market Researcher": "📈",
    "Product Manager": "📋",
    "Software Architect": "🏗️",
    "Frontend Developer": "🎨",
    "Backend Developer": "💻",
    "QA Engineer": "🧪",
    "Code Reviewer": "🔍",
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
            f"[bold]Project:[/bold] {project_name}\n"
            f"[bold]Model  :[/bold] {model}",
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


def print_agent_response(role: str, content: str) -> None:
    """Display an agent's response in a styled panel."""
    colour = _colour_for(role)
    emoji = _emoji_for(role)
    console.print(
        Panel(
            Markdown(content),
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
    console.print(Rule("[bold green]✅  All agents have completed their work[/bold green]", style="green"))
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
