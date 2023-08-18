"""Evaluation scoring functions and test case loader for the emojify eval suite."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from emojify.embeddings import get_embedding
from emojify.index import EmojiMatch


@dataclass
class EvalCase:
    """A single evaluation test case."""

    case_type: str  # "text_to_emoji" or "emoji_to_text"
    input_value: str  # The input text (for t2e) or emoji string (for e2t)
    expected: list[str]  # Expected emoji (t2e) or acceptable interpretations (e2t)
    category: str  # Semantic category (greeting, emotion, etc.)


@dataclass
class EvalResult:
    """Result of scoring a single eval case."""

    case: EvalCase
    score: int  # 1, 2, or 3
    output: str  # What the pipeline actually produced
    details: str  # Explanation of the score


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load evaluation test cases from a YAML file.

    Args:
        path: Path to the YAML file containing test cases.

    Returns:
        List of EvalCase instances, combining both text_to_emoji
        and emoji_to_text sections.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    cases: list[EvalCase] = []

    for entry in data.get("text_to_emoji", []):
        cases.append(
            EvalCase(
                case_type="text_to_emoji",
                input_value=entry["input_text"],
                expected=entry["expected_emoji"],
                category=entry["category"],
            )
        )

    for entry in data.get("emoji_to_text", []):
        cases.append(
            EvalCase(
                case_type="emoji_to_text",
                input_value=entry["input_emoji"],
                expected=entry["acceptable_interpretations"],
                category=entry["category"],
            )
        )

    return cases


def score_text_to_emoji(result: list[EmojiMatch], expected: list[str]) -> int:
    """Score a text-to-emoji result against expected emoji.

    Scoring rubric:
        3 - Any expected emoji appears in the top-3 results
        2 - Any expected emoji appears in the top-5 results (but not top-3)
        1 - None of the expected emoji appear in the top-5 results

    Args:
        result: Ranked list of EmojiMatch from text_to_emoji().
        expected: List of acceptable emoji characters.

    Returns:
        Integer score: 1, 2, or 3.
    """
    expected_set = set(expected)
    top_3_emoji = {m.emoji for m in result[:3]}
    top_5_emoji = {m.emoji for m in result[:5]}

    if expected_set & top_3_emoji:
        return 3
    if expected_set & top_5_emoji:
        return 2
    return 1


def score_emoji_to_text(
    interpretation: str,
    acceptable: list[str],
) -> int:
    """Score an emoji-to-text interpretation against acceptable answers.

    Uses cosine similarity between the embedding of the interpretation
    and the embeddings of each acceptable string. Takes the maximum
    similarity across all acceptable strings.

    Scoring rubric:
        3 - Max cosine similarity > 0.85
        2 - Max cosine similarity > 0.70
        1 - Max cosine similarity <= 0.70

    Args:
        interpretation: The LLM-generated interpretation string.
        acceptable: List of acceptable natural language interpretations.

    Returns:
        Integer score: 1, 2, or 3.
    """
    if not interpretation or not acceptable:
        return 1

    interp_embedding = get_embedding(interpretation)
    interp_norm = np.linalg.norm(interp_embedding)
    if interp_norm == 0:
        return 1

    max_similarity = 0.0
    for acc in acceptable:
        acc_embedding = get_embedding(acc)
        acc_norm = np.linalg.norm(acc_embedding)
        if acc_norm == 0:
            continue
        similarity = float(
            np.dot(interp_embedding, acc_embedding) / (interp_norm * acc_norm)
        )
        max_similarity = max(max_similarity, similarity)

    if max_similarity > 0.85:
        return 3
    if max_similarity > 0.70:
        return 2
    return 1
