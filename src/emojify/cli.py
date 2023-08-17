"""Click CLI definitions for emojify."""

import os
import re
import sys
import time

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from emojify import __version__

console = Console()

# Emoji detection regex — used by interactive mode to auto-detect input type
_EMOJI_CHARS_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U00002300-\U000023FF"
    "\U0001F3FB-\U0001F3FF"
    "]",
    flags=re.UNICODE,
)


def _is_emoji_input(text: str) -> bool:
    """Check if input is mostly emoji (>50% of non-whitespace characters)."""
    stripped = text.replace(" ", "")
    if not stripped:
        return False
    emoji_chars = _EMOJI_CHARS_RE.findall(stripped)
    return len(emoji_chars) / len(stripped) >= 0.5


def _validate_startup() -> None:
    """Check that required files and env vars exist before running commands."""
    from emojify.config import METADATA_PATH, INDEX_PATH

    if not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY environment variable is not set.\n"
            "Set it with: [bold]export OPENAI_API_KEY='your-key-here'[/bold]"
        )
        sys.exit(1)

    if not METADATA_PATH.exists() or not INDEX_PATH.exists():
        console.print(
            "[bold red]Error:[/bold red] Emoji index not found.\n"
            "Build it with: [bold]make build-index[/bold]"
        )
        sys.exit(1)


def _get_index(ctx: click.Context):
    """Lazy-load the EmojiIndex, caching it in Click context."""
    if "index" not in ctx.obj:
        from emojify.index import EmojiIndex

        ctx.obj["index"] = EmojiIndex()
    return ctx.obj["index"]


@click.group()
@click.pass_context
def cli(ctx):
    """Bidirectional emoji-text translator."""
    ctx.ensure_object(dict)


@cli.command()
def version():
    """Print the emojify version."""
    click.echo(f"emojify v{__version__}")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results to return.")
@click.option("--no-diversity", is_flag=True, help="Disable diversity filtering in suggested sequence.")
@click.option("--verbose", is_flag=True, help="Show timing information.")
@click.pass_context
def text(ctx, query, top_k, no_diversity, verbose):
    """Find the best emoji for a text query.

    Example: emojify text "just deployed to production at 2am"
    """
    _validate_startup()
    index = _get_index(ctx)

    from emojify.text_to_emoji import text_to_emoji

    t0 = time.time()
    try:
        results = text_to_emoji(query, index, top_k=top_k)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    search_time = time.time() - t0

    # Build Rich table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Emoji", width=6, justify="center")
    table.add_column("Score", width=8, justify="right")
    table.add_column("Name", min_width=15)
    table.add_column("Keywords", min_width=20)

    for r in results:
        keywords_str = ", ".join(r.keywords[:5])
        table.add_row(
            r.emoji,
            f"{r.score:.2f}",
            r.short_name,
            keywords_str,
        )

    console.print(table)

    # Suggested sequence (diversity-filtered top 3)
    if not no_diversity:
        diverse_results = text_to_emoji(
            query, index, top_k=top_k * 2, diverse=True, target_count=3,
        )
        sequence = "".join(r.emoji for r in diverse_results)
        console.print(f"\nSuggested sequence: [bold]{sequence}[/bold]")

    if verbose:
        console.print(f"\n[dim]Search time: {search_time:.3f}s[/dim]")


@cli.command()
@click.argument("query")
@click.option("--count", default=5, help="Number of emoji to return.")
@click.option("--verbose", is_flag=True, help="Show timing information.")
@click.pass_context
def suggest(ctx, query, count, verbose):
    """Get a quick emoji sequence for a text query.

    Returns just the emoji characters — no table, no scores.

    Example: emojify suggest "good morning"
    """
    _validate_startup()
    index = _get_index(ctx)

    from emojify.text_to_emoji import suggest as suggest_fn

    t0 = time.time()
    try:
        result = suggest_fn(query, index, count=count)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    elapsed = time.time() - t0

    click.echo(result)

    if verbose:
        console.print(f"[dim]Time: {elapsed:.3f}s[/dim]")


@cli.command()
@click.argument("emoji_string")
@click.option("--no-llm", is_flag=True, help="Skip LLM interpretation (Stage 1 only).")
@click.option("--verbose", is_flag=True, help="Show timing information.")
@click.pass_context
def decode(ctx, emoji_string, no_llm, verbose):
    """Decode an emoji string into natural language.

    Example: emojify decode "🎉🍕🎂"
    """
    _validate_startup()
    index = _get_index(ctx)

    from emojify.decoder import decode_emoji

    t0 = time.time()
    try:
        result = decode_emoji(emoji_string, index, use_llm=not no_llm)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    elapsed = time.time() - t0

    if not result.individual:
        console.print("[yellow]No emoji found in input.[/yellow]")
        return

    # Individual meanings
    console.print("\n[bold]Individual meanings:[/bold]")
    for desc in result.individual:
        keywords_str = ", ".join(desc.keywords[:5])
        if desc.found_in_index:
            console.print(f"  {desc.emoji}  {desc.short_name} — {keywords_str}")
        else:
            console.print(f"  {desc.emoji}  [dim]{desc.short_name}[/dim]")

    # Combined interpretation
    if result.combined_interpretation:
        console.print(
            f"\n[bold]Combined interpretation:[/bold] "
            f"[green]\"{result.combined_interpretation}\"[/green]"
        )

    if verbose:
        console.print(f"\n[dim]Time: {elapsed:.3f}s[/dim]")


@cli.command()
@click.option("--verbose", is_flag=True, help="Show timing information.")
@click.pass_context
def interactive(ctx, verbose):
    """Start an interactive REPL session.

    Type text to get emoji suggestions, or paste emoji to decode them.
    Type 'q' or 'quit' to exit.
    """
    _validate_startup()
    index = _get_index(ctx)

    from emojify.text_to_emoji import suggest as suggest_fn
    from emojify.decoder import decode_emoji

    console.print(
        f"[bold cyan]emojify v{__version__}[/bold cyan] — "
        "type text or paste emoji ([bold]q[/bold] to quit)\n"
    )

    while True:
        try:
            user_input = console.input("[bold]> [/bold]")
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye! 👋")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("q", "quit"):
            console.print("Goodbye! 👋")
            break

        t0 = time.time()
        try:
            if _is_emoji_input(user_input):
                # Decode mode
                result = decode_emoji(user_input, index, use_llm=True)
                if result.combined_interpretation:
                    console.print(
                        f"[green]\"{result.combined_interpretation}\"[/green]\n"
                    )
                else:
                    for desc in result.individual:
                        console.print(f"  {desc.emoji}  {desc.short_name}")
                    console.print()
            else:
                # Suggest mode
                emoji_str = suggest_fn(user_input, index, count=5)
                console.print(f"{emoji_str}\n")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}\n")

        if verbose:
            elapsed = time.time() - t0
            console.print(f"[dim]Time: {elapsed:.3f}s[/dim]\n")
