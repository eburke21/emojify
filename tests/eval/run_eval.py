"""Evaluation runner for emojify — runs all 50 eval cases and scores results.

Usage:
    python tests/eval/run_eval.py
    # or: make eval

Requires:
    - OPENAI_API_KEY environment variable set
    - Built emoji index (make build-index)
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emojify.eval import (
    EvalCase,
    EvalResult,
    load_eval_cases,
    score_text_to_emoji,
    score_emoji_to_text,
)
from emojify.index import EmojiIndex
from emojify.text_to_emoji import text_to_emoji
from emojify.decoder import decode_emoji

console = Console()

EVAL_CASES_PATH = Path(__file__).resolve().parent / "test_cases.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_eval() -> list[EvalResult]:
    """Run the full evaluation suite.

    Loads the real index, runs both pipelines against all test cases,
    scores the results, and returns the list of EvalResult objects.
    """
    console.print("[bold cyan]Loading emoji index...[/bold cyan]")
    index = EmojiIndex()

    console.print("[bold cyan]Loading eval cases...[/bold cyan]")
    cases = load_eval_cases(EVAL_CASES_PATH)
    console.print(f"Loaded {len(cases)} eval cases.\n")

    results: list[EvalResult] = []

    # --- Text to Emoji cases ---
    t2e_cases = [c for c in cases if c.case_type == "text_to_emoji"]
    console.print(
        f"[bold]Running {len(t2e_cases)} text_to_emoji cases...[/bold]"
    )

    for case in t2e_cases:
        try:
            matches = text_to_emoji(case.input_value, index, top_k=5)
            score = score_text_to_emoji(matches, case.expected)
            output = " ".join(
                f"{m.emoji}({m.score:.2f})" for m in matches[:5]
            )
            top_emoji = [m.emoji for m in matches[:5]]
            details = (
                f"Expected any of {case.expected} in top-5. Got: {top_emoji}"
            )
        except Exception as e:
            score = 1
            output = f"ERROR: {e}"
            details = str(e)

        results.append(
            EvalResult(case=case, score=score, output=output, details=details)
        )
        marker = {3: "[green]✓[/green]", 2: "[yellow]~[/yellow]", 1: "[red]✗[/red]"}[score]
        console.print(
            f"  {marker} score={score} | "
            f"\"{case.input_value}\" → {output[:60]}"
        )

    # --- Emoji to Text cases ---
    e2t_cases = [c for c in cases if c.case_type == "emoji_to_text"]
    console.print(
        f"\n[bold]Running {len(e2t_cases)} emoji_to_text cases...[/bold]"
    )

    for case in e2t_cases:
        try:
            decode_result = decode_emoji(
                case.input_value, index, use_llm=True
            )
            interpretation = decode_result.combined_interpretation
            score = score_emoji_to_text(interpretation, case.expected)
            output = interpretation
            details = (
                f"Interpretation: \"{interpretation}\". "
                f"Acceptable: {case.expected}"
            )
        except Exception as e:
            score = 1
            output = f"ERROR: {e}"
            details = str(e)

        results.append(
            EvalResult(case=case, score=score, output=output, details=details)
        )
        marker = {3: "[green]✓[/green]", 2: "[yellow]~[/yellow]", 1: "[red]✗[/red]"}[score]
        console.print(
            f"  {marker} score={score} | "
            f"{case.input_value} → \"{output[:60]}\""
        )

    return results


def save_results(results: list[EvalResult]) -> Path:
    """Save eval results to results/eval_run.json."""
    RESULTS_DIR.mkdir(exist_ok=True)

    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Per-category averages
    category_scores: dict[str, list[int]] = defaultdict(list)
    for r in results:
        key = f"{r.case.case_type}/{r.case.category}"
        category_scores[key].append(r.score)

    category_avgs = {
        k: round(sum(v) / len(v), 3)
        for k, v in sorted(category_scores.items())
    }

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(results),
        "average_score": round(avg_score, 3),
        "target_score": 2.3,
        "meets_target": avg_score >= 2.3,
        "category_averages": category_avgs,
        "cases": [
            {
                "case_type": r.case.case_type,
                "input": r.case.input_value,
                "expected": r.case.expected,
                "category": r.case.category,
                "score": r.score,
                "output": r.output,
                "details": r.details,
            }
            for r in results
        ],
    }

    out_path = RESULTS_DIR / "eval_run.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return out_path


def print_summary(results: list[EvalResult]) -> None:
    """Print a Rich summary table of eval results."""
    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    console.print("\n" + "=" * 60)
    console.print("[bold cyan]EVALUATION SUMMARY[/bold cyan]")
    console.print("=" * 60)

    # Overall stats
    score_dist = {1: 0, 2: 0, 3: 0}
    for s in scores:
        score_dist[s] += 1

    target_met = (
        "[green]YES[/green]" if avg_score >= 2.3 else "[red]NO[/red]"
    )
    console.print(f"\nOverall average: [bold]{avg_score:.2f}[/bold] / 3.00")
    console.print(f"Target (>= 2.3): {target_met}")
    console.print(
        f"Distribution: "
        f"[green]3={score_dist[3]}[/green] | "
        f"[yellow]2={score_dist[2]}[/yellow] | "
        f"[red]1={score_dist[1]}[/red]"
    )

    # Per-category table
    category_scores: dict[str, list[int]] = defaultdict(list)
    for r in results:
        key = f"{r.case.case_type}/{r.case.category}"
        category_scores[key].append(r.score)

    table = Table(
        title="Per-Category Averages",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Pipeline / Category", min_width=30)
    table.add_column("Avg Score", justify="right", width=10)
    table.add_column("Cases", justify="right", width=8)

    for key in sorted(category_scores.keys()):
        vals = category_scores[key]
        avg = sum(vals) / len(vals)
        color = "green" if avg >= 2.5 else ("yellow" if avg >= 2.0 else "red")
        table.add_row(
            key, f"[{color}]{avg:.2f}[/{color}]", str(len(vals))
        )

    console.print()
    console.print(table)

    # Worst 5 cases
    sorted_results = sorted(results, key=lambda r: (r.score, r.output))
    worst_5 = sorted_results[:5]

    console.print("\n[bold red]Worst 5 Cases:[/bold red]")
    for r in worst_5:
        console.print(
            f"  score={r.score} | [{r.case.case_type}] "
            f"\"{r.case.input_value}\" → \"{r.output[:80]}\""
        )
        console.print(f"           Expected: {r.case.expected}")

    console.print()


def main():
    """Entry point for the eval runner."""
    console.print("[bold]emojify Evaluation Suite[/bold]\n")

    t0 = time.time()
    results = run_eval()
    elapsed = time.time() - t0

    out_path = save_results(results)
    print_summary(results)

    console.print(f"Results saved to: [bold]{out_path}[/bold]")
    console.print(f"Total time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
