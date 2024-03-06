"""Tests for the text-to-emoji pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from emojify.index import EmojiIndex, EmojiMatch
from emojify.text_to_emoji import text_to_emoji, suggest


def _create_test_index(n: int = 10, dim: int = 1536) -> EmojiIndex:
    """Create a small fake EmojiIndex for testing."""
    tmpdir = tempfile.mkdtemp()
    metadata_path = Path(tmpdir) / "metadata.json"
    index_path = Path(tmpdir) / "index.npz"

    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, dim).astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    categories = [
        "Smileys & Emotion",
        "People & Body",
        "Animals & Nature",
        "Food & Drink",
        "Travel & Places",
        "Activities",
        "Objects",
        "Symbols",
        "Flags",
        "Smileys & Emotion",
    ]
    emoji_chars = [chr(0x1F600 + i) for i in range(n)]
    metadata = [
        {
            "emoji": emoji_chars[i],
            "short_name": f"emoji_{i}",
            "keywords": [f"kw_{i}_a", f"kw_{i}_b"],
            "category": categories[i % len(categories)],
            "description": f"Emoji {i} description",
        }
        for i in range(n)
    ]

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    np.savez(
        index_path,
        embeddings=embeddings,
        emoji_list=np.array(emoji_chars, dtype=object),
    )

    return EmojiIndex(metadata_path=metadata_path, index_path=index_path)


class TestTextToEmoji:
    """Tests for the text_to_emoji function."""

    def setup_method(self):
        self.index = _create_test_index(n=10)

    @patch("emojify.text_to_emoji.get_embedding")
    def test_basic_search(self, mock_embed):
        """text_to_emoji returns results with scores in descending order."""
        # Use a known vector that will produce different similarities
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        results = text_to_emoji("happy times", self.index, top_k=5)

        assert len(results) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        for r in results:
            assert isinstance(r, EmojiMatch)

    @patch("emojify.text_to_emoji.get_embedding")
    def test_top_k_respected(self, mock_embed):
        """Requesting top_k=3 returns exactly 3 results."""
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        results = text_to_emoji("test query", self.index, top_k=3)

        assert len(results) == 3

    @patch("emojify.text_to_emoji.get_embedding")
    def test_results_have_expected_fields(self, mock_embed):
        """Each result is an EmojiMatch with all expected fields."""
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        results = text_to_emoji("some query", self.index, top_k=1)

        assert len(results) == 1
        match = results[0]
        assert isinstance(match.emoji, str)
        assert isinstance(match.score, float)
        assert isinstance(match.short_name, str)
        assert isinstance(match.keywords, list)

    @patch("emojify.text_to_emoji.get_embedding")
    def test_suggest_returns_string(self, mock_embed):
        """suggest() returns a plain string of emoji characters."""
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        result = suggest("happy morning", self.index, count=3)

        assert isinstance(result, str)
        assert len(result) >= 1  # At least 1 emoji character
        # Every non-space character should be a high Unicode character (emoji range)
        for char in result:
            assert ord(char) > 127 or char == " "

    @patch("emojify.text_to_emoji.get_embedding")
    def test_suggest_count(self, mock_embed):
        """suggest() returns the requested number of emoji."""
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        result = suggest("test", self.index, count=4)

        # Space-separated emoji, so split to count
        assert len(result.split()) == 4
