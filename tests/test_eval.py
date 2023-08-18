"""Tests for the evaluation scoring functions and YAML loader."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from emojify.index import EmojiMatch
from emojify.eval import (
    EvalCase,
    load_eval_cases,
    score_text_to_emoji,
    score_emoji_to_text,
)


def _make_match(emoji: str, score: float) -> EmojiMatch:
    """Helper to build a test EmojiMatch."""
    return EmojiMatch(
        emoji=emoji, score=score, short_name=f"name_{emoji}", keywords=[]
    )


class TestScoreTextToEmoji:
    """Tests for score_text_to_emoji scoring function."""

    def test_score_3_when_expected_in_top_3(self):
        results = [
            _make_match("🎂", 0.9),
            _make_match("🎉", 0.85),
            _make_match("🥳", 0.80),
            _make_match("🍕", 0.75),
            _make_match("🚀", 0.70),
        ]
        assert score_text_to_emoji(results, ["🎂", "🎉", "🥳"]) == 3

    def test_score_3_with_single_expected_in_top_3(self):
        results = [
            _make_match("🎂", 0.9),
            _make_match("😀", 0.85),
            _make_match("🍕", 0.80),
            _make_match("🚀", 0.75),
            _make_match("💻", 0.70),
        ]
        assert score_text_to_emoji(results, ["🎂", "🎉", "🥳"]) == 3

    def test_score_2_when_expected_in_top_5_not_top_3(self):
        results = [
            _make_match("😀", 0.9),
            _make_match("🍕", 0.85),
            _make_match("🚀", 0.80),
            _make_match("🎂", 0.75),
            _make_match("💻", 0.70),
        ]
        assert score_text_to_emoji(results, ["🎂", "🎉", "🥳"]) == 2

    def test_score_1_when_expected_not_in_top_5(self):
        results = [
            _make_match("😀", 0.9),
            _make_match("🍕", 0.85),
            _make_match("🚀", 0.80),
            _make_match("💻", 0.75),
            _make_match("🎮", 0.70),
        ]
        assert score_text_to_emoji(results, ["🎂", "🎉", "🥳"]) == 1

    def test_score_with_empty_results(self):
        assert score_text_to_emoji([], ["🎂"]) == 1

    def test_score_with_fewer_than_5_results(self):
        results = [
            _make_match("😀", 0.9),
            _make_match("🎂", 0.85),
        ]
        assert score_text_to_emoji(results, ["🎂"]) == 3


class TestScoreEmojiToText:
    """Tests for score_emoji_to_text scoring function."""

    @patch("emojify.eval.get_embedding")
    def test_score_3_high_similarity(self, mock_embed):
        """Identical embeddings -> cosine sim = 1.0 -> score 3."""
        vec = np.ones(1536) / np.sqrt(1536)
        mock_embed.return_value = vec.copy()
        assert score_emoji_to_text("Happy birthday", ["Happy birthday!"]) == 3

    @patch("emojify.eval.get_embedding")
    def test_score_1_low_similarity(self, mock_embed):
        """Orthogonal embeddings -> cosine sim ~ 0.0 -> score 1."""
        rng = np.random.RandomState(42)
        vec1 = rng.randn(1536)
        vec1 = vec1 / np.linalg.norm(vec1)
        # Make vec2 orthogonal to vec1
        vec2 = rng.randn(1536)
        vec2 = vec2 - np.dot(vec2, vec1) * vec1
        vec2 = vec2 / np.linalg.norm(vec2)

        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            return vec1.copy() if call_count[0] == 1 else vec2.copy()

        mock_embed.side_effect = side_effect
        assert score_emoji_to_text("something", ["completely different"]) == 1

    @patch("emojify.eval.get_embedding")
    def test_takes_max_across_acceptable(self, mock_embed):
        """Uses the maximum similarity across all acceptable strings."""
        rng = np.random.RandomState(42)
        vec_interp = rng.randn(1536)
        vec_interp = vec_interp / np.linalg.norm(vec_interp)

        # Make one acceptable very similar, another orthogonal
        vec_close = vec_interp.copy()
        vec_far = rng.randn(1536)
        vec_far = vec_far - np.dot(vec_far, vec_interp) * vec_interp
        vec_far = vec_far / np.linalg.norm(vec_far)

        calls = [0]

        def side_effect(text):
            calls[0] += 1
            if calls[0] == 1:
                return vec_interp.copy()
            elif calls[0] == 2:
                return vec_far.copy()
            else:
                return vec_close.copy()

        mock_embed.side_effect = side_effect
        # Should pick max similarity (from vec_close) -> score 3
        assert score_emoji_to_text("test", ["far away", "very close"]) == 3

    @patch("emojify.eval.get_embedding")
    def test_score_2_medium_similarity(self, mock_embed):
        """Cosine sim between 0.70 and 0.85 -> score 2."""
        rng = np.random.RandomState(42)
        base = rng.randn(1536)
        base = base / np.linalg.norm(base)
        # Construct a vector with ~0.78 cosine similarity:
        # modified = 0.78 * base + sqrt(1 - 0.78^2) * orthogonal
        ortho = rng.randn(1536)
        ortho = ortho - np.dot(ortho, base) * base
        ortho = ortho / np.linalg.norm(ortho)
        target_sim = 0.78
        modified = target_sim * base + np.sqrt(1 - target_sim**2) * ortho
        modified = modified / np.linalg.norm(modified)

        sim = float(np.dot(base, modified))
        assert 0.70 < sim <= 0.85, f"Similarity {sim} not in expected range"

        calls = [0]

        def side_effect(text):
            calls[0] += 1
            return base.copy() if calls[0] == 1 else modified.copy()

        mock_embed.side_effect = side_effect
        assert score_emoji_to_text("interp", ["acceptable"]) == 2

    def test_empty_interpretation(self):
        assert score_emoji_to_text("", ["some text"]) == 1

    def test_empty_acceptable(self):
        assert score_emoji_to_text("some text", []) == 1


class TestLoadEvalCases:
    """Tests for the YAML loader."""

    def test_load_eval_cases(self):
        cases_path = Path(__file__).resolve().parent / "eval" / "test_cases.yaml"
        cases = load_eval_cases(cases_path)

        assert len(cases) == 50

        t2e_cases = [c for c in cases if c.case_type == "text_to_emoji"]
        e2t_cases = [c for c in cases if c.case_type == "emoji_to_text"]
        assert len(t2e_cases) == 30
        assert len(e2t_cases) == 20

        for case in cases:
            assert case.input_value, f"Empty input_value: {case}"
            assert len(case.expected) >= 2, f"Expected needs >=2 options: {case}"
            assert case.category, f"Empty category: {case}"

    def test_eval_case_dataclass(self):
        case = EvalCase(
            case_type="text_to_emoji",
            input_value="hello",
            expected=["👋", "😊"],
            category="greeting",
        )
        assert case.case_type == "text_to_emoji"
        assert case.input_value == "hello"
        assert len(case.expected) == 2
