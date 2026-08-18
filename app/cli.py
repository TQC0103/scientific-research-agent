import json
import urllib.error
import urllib.request
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from app.agent.graph import research_graph
from app.config import settings
from app.db.database import search_local
from app.ingestion.indexing import index_paper
from app.retrieval.vector_store import retrieve
from app.tools.arxiv_search import search_arxiv

app = typer.Typer(no_args_is_help=True, help="Local-first scientific research assistant")
console = Console()


@app.command()
def doctor() -> None:
    """Check directories, Ollama, and required local models."""
    settings.ensure_directories()
    console.print(f"[green]OK[/green] data directory: {settings.data_dir}")
    try:
        with urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=5) as response:
            payload = json.load(response)
        names = {model["name"] for model in payload.get("models", [])}
        for model in (settings.ollama_model, settings.ollama_embed_model):
            present = model in names or any(name.startswith(f"{model}:") for name in names)
            console.print(
                f"{'[green]OK[/green]' if present else '[red]MISSING[/red]'} model: {model}"
            )
    except (OSError, urllib.error.URLError) as exc:
        console.print(f"[red]Ollama unavailable:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command("search")
def search_command(
    query: str,
    year: Annotated[int | None, typer.Option()] = None,
    category: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1, max=50)] = 10,
    date_mode: Annotated[
        str,
        typer.Option(help="first_submitted, last_revised, or active_in_period"),
    ] = "first_submitted",
) -> None:
    """Search arXiv and save metadata to SQLite."""
    valid_modes = {"first_submitted", "last_revised", "active_in_period"}
    if date_mode not in valid_modes:
        raise typer.BadParameter(f"date-mode must be one of: {', '.join(sorted(valid_modes))}")
    papers = search_arxiv(
        query,
        year=year,
        category=category,
        max_results=limit,
        date_mode=date_mode,  # type: ignore[arg-type]
    )
    table = Table("arXiv version", "First submitted", "Last revised", "Title")
    for paper in papers:
        table.add_row(
            paper.get("versioned_id") or paper["arxiv_id"],
            (paper.get("first_submitted_at") or "")[:10],
            (paper.get("last_revised_at") or "")[:10],
            paper["title"],
        )
    console.print(table)


@app.command("index")
def index_command(arxiv_id: str, force_download: bool = False) -> None:
    """Download, parse, chunk, embed, and index one paper."""
    destination, count = index_paper(arxiv_id, force_download=force_download)
    console.print(f"[green]Indexed[/green] {arxiv_id}: {count} chunks -> {destination}")


@app.command("retrieve")
def retrieve_command(arxiv_id: str, query: str, top_k: int = 5) -> None:
    """Retrieve evidence without invoking the reasoning model."""
    for item in retrieve(arxiv_id, query, top_k=top_k):
        console.rule(f"p.{item['page']} · {item['section']} · score={item['score']:.3f}")
        console.print(item["text"])


@app.command("local-search")
def local_search_command(query: str, limit: int = 10) -> None:
    """Search cached title/abstract metadata with SQLite FTS5."""
    table = Table("arXiv version", "First submitted", "Last revised", "Title")
    for paper in search_local(query, limit=limit):
        table.add_row(
            paper.get("versioned_id") or paper["arxiv_id"],
            (paper.get("first_submitted_at") or "")[:10],
            (paper.get("last_revised_at") or "")[:10],
            paper["title"],
        )
    console.print(table)


@app.command("ask")
def ask_command(
    query: str, paper_id: Annotated[list[str], typer.Option("--paper-id", "-p")]
) -> None:
    """Answer from one or more already-indexed papers."""
    result = research_graph.invoke({"user_query": query, "paper_ids": paper_id})
    console.print(result["answer"])


@app.command("chat")
def chat_command(
    query: str,
    paper_id: Annotated[list[str] | None, typer.Option("--paper-id", "-p")] = None,
    trace: Annotated[bool, typer.Option(help="Show discovery and retrieval decisions")] = False,
) -> None:
    """Run the bounded LangGraph workflow, including lazy search/index."""
    result = research_graph.invoke({"user_query": query, "paper_ids": paper_id or []})
    if trace:
        attempts = result.get("retrieval_attempt_counts", {})
        coverage = result.get("evidence_verification", {}).get("coverage_mode", "any")
        console.print(
            f"[dim]discovery={result.get('discovery_source')} "
            f"selected={result.get('selected_papers', [])} "
            f"failed={result.get('failed_papers', [])} "
            f"coverage={coverage} "
            f"retrieval_attempts={attempts} "
            f"verified={result.get('evidence_sufficient', False)}[/dim]"
        )
    console.print(result["answer"])


@app.command("ui")
def ui_command(share: bool = False) -> None:
    """Start the local Gradio interface."""
    from ui.gradio_app import demo

    demo.launch(share=share)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
