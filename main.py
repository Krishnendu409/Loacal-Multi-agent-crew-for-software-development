#!/usr/bin/env python3
"""Local Multi-Agent Software Development Crew – CLI entry point.

Usage examples
--------------
Run interactively (will prompt for project name and requirements):

    python main.py run

Provide project details inline:

    python main.py run --project "Todo API" --requirements "Build a REST API for a todo app"

Override the Ollama model for this run:

    python main.py run --model llama3.2 --project "Todo API"

Use a custom config file:

    python main.py run --config my_config.yaml

List available Ollama models:

    python main.py models

Show the current configuration:

    python main.py config
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.pretty import pprint
from rich.prompt import Confirm

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path so `src.*` imports work regardless
# of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agents.definitions import build_agents
from src.config.settings import load_config
from src.crew.dev_crew import DevCrew
from src.utils import display
from src.utils.ollama_client import OllamaClient

app = typer.Typer(
    name="dev-crew",
    help="🤖 Local Multi-Agent Software Development Crew – free, local, fast.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Short project name (used for output filenames)."
    ),
    requirements: Optional[str] = typer.Option(
        None,
        "--requirements",
        "-r",
        help="Raw problem statement / requirements. If omitted you will be prompted.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Ollama model to use (overrides config.yaml)."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a custom config.yaml file."
    ),
    auto_approve_strategy: bool = typer.Option(
        False,
        "--auto-approve-strategy",
        help="Skip strategy confirmation prompt and proceed automatically.",
    ),
) -> None:
    """Start the multi-agent development crew on a new project."""
    display.print_banner()

    cfg = load_config(config)
    allowed_models = {
        m.strip()
        for m in cfg["llm"].get("allowed_models", [])
        if isinstance(m, str) and m.strip()
    }

    # --- Resolve model -------------------------------------------------
    effective_model = model or cfg["llm"]["model"]
    _validate_allowed_model(effective_model, allowed_models, "Selected model")

    # --- Gather project name ------------------------------------------
    if not project:
        project = typer.prompt("📌 Project name")

    # --- Gather requirements ------------------------------------------
    if not requirements:
        console.print(
            "\n[bold cyan]📝 Enter your problem statement (or requirements).[/bold cyan]\n"
            "  You can type multiple lines.  When finished, type a single "
            "[bold]END[/bold] on its own line and press Enter.\n"
        )
        lines: list[str] = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        requirements = "\n".join(lines)

    if not requirements.strip():
        display.print_error("Requirements cannot be empty.  Exiting.")
        raise typer.Exit(code=1)

    display.print_project_start(project, effective_model)

    # --- Build LLM client --------------------------------------------
    llm = OllamaClient(
        model=effective_model,
        base_url=cfg["llm"]["base_url"],
        options=cfg["llm"].get("options", {}),
        retries=int(cfg["llm"].get("retries", 1)),
        timeout_seconds=cfg["llm"].get("timeout_seconds"),
    )

    # --- Build agents -------------------------------------------------
    llm_for_agents = dict(cfg["llm"])
    if model:
        _validate_allowed_model(model, allowed_models, "CLI model override")
        llm_for_agents["routing"] = {k: model for k in cfg["agents"].keys()}
        llm_for_agents["fallbacks"] = {}
    agents = build_agents(
        llm,
        enabled=cfg["agents"],
        llm_config=llm_for_agents,
        skills_config=cfg.get("skills", {}),
    )
    if not agents:
        display.print_error(
            "No agents are enabled.  Check the 'agents' section in config.yaml."
        )
        raise typer.Exit(code=1)

    # --- Run crew -----------------------------------------------------
    crew = DevCrew(
        agents=agents,
        output_dir=cfg["output"]["directory"],
        save_individual=cfg["output"]["save_individual_responses"],
        save_report=cfg["output"]["save_final_report"],
        max_fix_iterations=int(cfg.get("crew", {}).get("max_fix_iterations", 1)),
        stop_on_no_major_issues=bool(
            cfg.get("crew", {}).get("stop_on_no_major_issues", True)
        ),
    )

    try:
        require_gate = bool(cfg.get("crew", {}).get("require_strategy_approval", True))

        def _confirm_strategy(_: dict[str, str]) -> bool:
            if auto_approve_strategy:
                return True
            console.print(
                "\n[bold yellow]Strategy phase complete.[/bold yellow] "
                "Proceed to implementation?"
            )
            return Confirm.ask("Continue", default=False)

        crew.kickoff_with_strategy_gate(
            requirements=requirements,
            project_name=project,
            require_strategy_approval=require_gate,
            strategy_approval_callback=_confirm_strategy if require_gate else None,
        )
    except Exception as exc:  # noqa: BLE001
        display.print_error(str(exc))
        _hint_common_errors(exc)
        raise typer.Exit(code=1) from exc


@app.command()
def models() -> None:
    """List the Ollama models available on this machine."""
    try:
        import ollama  # type: ignore[import-untyped]

        cfg = load_config()
        client = ollama.Client(host=cfg["llm"]["base_url"])
        model_list = client.list()
        if not model_list.get("models"):
            console.print("[yellow]No models found.  Pull one with:[/yellow]  ollama pull mistral")
            return
        console.print("\n[bold cyan]Available Ollama models:[/bold cyan]")
        for m in model_list["models"]:
            name = m.get("name") or m.get("model", "unknown")
            size_gb = m.get("size", 0) / 1_073_741_824
            console.print(f"  • [green]{name}[/green]  ({size_gb:.1f} GB)")
        console.print()
    except ImportError:
        display.print_error("'ollama' package not installed.  Run: pip install ollama")
    except Exception as exc:  # noqa: BLE001
        display.print_error(f"Could not connect to Ollama: {exc}")
        console.print("  Make sure the Ollama daemon is running: [bold]ollama serve[/bold]")


@app.command(name="config")
def show_config(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file.")
) -> None:
    """Display the resolved configuration (merged defaults + config.yaml)."""
    cfg = load_config(config)
    console.print("\n[bold cyan]Resolved configuration:[/bold cyan]\n")
    pprint(cfg)
    console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hint_common_errors(exc: Exception) -> None:
    msg = str(exc).lower()
    if "connection" in msg or "connect" in msg or "refused" in msg:
        console.print(
            "\n[yellow]💡 Tip:[/yellow] Could not reach the Ollama daemon.  "
            "Start it with:  [bold]ollama serve[/bold]"
        )
    elif "model" in msg and ("not found" in msg or "pull" in msg):
        console.print(
            "\n[yellow]💡 Tip:[/yellow] The model was not found locally.  "
            "Pull it with:  [bold]ollama pull qwen2.5:7b-instruct[/bold]"
        )


def _validate_allowed_model(model_name: str, allowed_models: set[str], context: str) -> None:
    if allowed_models and model_name not in allowed_models:
        display.print_error(
            f"{context} is not allowed by config.llm.allowed_models. "
            f"Allowed: {sorted(allowed_models)}"
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
