#!/usr/bin/env python3
"""CLI entry point for p4harness agentic network management."""
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from agents import pipeline, swap_model

app = typer.Typer(help="p4harness — LLM-agentic P4 network change management")
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Natural language description of the network change"),
    p4_file: str = typer.Option(None, "--p4", help="Existing P4 file to edit"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run pipeline but do not write output"),
):
    """Run the full planner → coder → reviewer → tester → gate pipeline."""
    initial_p4 = ""
    if p4_file:
        with open(p4_file) as f:
            initial_p4 = f.read()

    console.print(Panel(f"[bold cyan]Task:[/] {task}", title="p4harness"))

    state = {
        "task": task,
        "p4_code": initial_p4,
        "plan": "",
        "review_notes": [],
        "test_output": "",
        "review_passed": False,
        "tests_passed": False,
        "approved_for_push": False,
        "round": 0,
    }

    for step in pipeline.stream(state):
        node_name = list(step.keys())[0]
        node_state = step[node_name]
        console.print(f"\n[bold green]→ {node_name}[/]")

        if "plan" in node_state and node_state["plan"]:
            console.print(Panel(node_state["plan"], title="Plan", border_style="blue"))

        if "p4_code" in node_state and node_state["p4_code"]:
            syntax = Syntax(node_state["p4_code"], "p4", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="P4 Code", border_style="yellow"))

        if "review_notes" in node_state and node_state["review_notes"]:
            for note in node_state["review_notes"]:
                color = "green" if "PASS" in note.upper() else "red"
                console.print(Panel(note, title="Review", border_style=color))

        if "test_output" in node_state and node_state["test_output"]:
            console.print(Panel(node_state["test_output"], title="Tests", border_style="magenta"))

        if "approved_for_push" in node_state:
            status = "[bold green]APPROVED[/]" if node_state["approved_for_push"] else "[bold red]BLOCKED[/]"
            console.print(f"\nFinal verdict: {status}")

            if node_state["approved_for_push"] and not dry_run and "p4_code" in node_state:
                out_path = "output.p4"
                with open(out_path, "w") as f:
                    f.write(node_state.get("p4_code", ""))
                console.print(f"[green]P4 written to {out_path}[/]")


@app.command()
def swap(
    role: str = typer.Argument(..., help="Agent role: planner, p4_coder, reviewer, tester"),
    model: str = typer.Argument(..., help="Ollama model name (e.g. qwen2.5-coder:32b)"),
):
    """Hot-swap the model for an agent role."""
    swap_model(role, model)
    console.print(f"[green]Swapped {role} → {model}[/]")


if __name__ == "__main__":
    app()
