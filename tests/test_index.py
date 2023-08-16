"""Tests for the EmojiIndex class."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from emojify.index import EmojiIndex, EmojiMatch


def _create_test_index(n: int = 10, dim: int = 1536) -> tuple[Path, Path]:
    """Create a small fake index and metadata for testing.

    Returns (metadata_path, index_path).
    """
    tmpdir = tempfile.mkdtemp()
    metadata_path = Path(tmpdir) / "metadata.json"
    index_path = Path(tmpdir) / "index.npz"

    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, dim).astype(np.float64)
    # Normalize so cosine similarity is meaningful
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    emoji_chars = [chr(0x1F600 + i) for i in range(n)]
    metadata = [
        {
            "emoji": emoji_chars[i],
            "short_name": f"emoji_{i}",
            "keywords": [f"kw_{i}_a", f"kw_{i}_b"],
            "category": "Smileys & Emotion",
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

    return metadata_path, index_path


class TestEmojiIndex:
    """Tests for EmojiIndex load and search."""

    def setup_method(self):
        self.metadata_path, self.index_path = _create_test_index(n=10)
        self.index = EmojiIndex(
            metadata_path=self.metadata_path,
            index_path=self.index_path,
        )

    def test_load_index(self):
        """Index loads with correct shape and metadata count."""
        assert self.index.embeddings.shape == (10, 1536)
        assert len(self.index.metadata) == 10
        assert len(self.index.emoji_to_idx) == 10

    def test_search_returns_sorted(self):
        """Search results are sorted by descending similarity score."""
        query = np.random.randn(1536)
        results = self.index.search(query, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k(self):
        """Requesting top_k=3 returns exactly 3 results."""
        query = np.random.randn(1536)
        results = self.index.search(query, top_k=3)
        assert len(results) == 3

    def test_search_returns_emoji_match(self):
        """Each result is an EmojiMatch with expected fields."""
        query = np.random.randn(1536)
        results = self.index.search(query, top_k=1)
        assert len(results) == 1
        match = results[0]
        assert isinstance(match, EmojiMatch)
        assert isinstance(match.emoji, str)
        assert isinstance(match.score, float)
        assert isinstance(match.short_name, str)
        assert isinstance(match.keywords, list)

    def test_search_top_result_is_self(self):
        """Searching with an emoji's own embedding returns that emoji first."""
        # Use the first emoji's embedding as query
        query = self.index.embeddings[0]
        results = self.index.search(query, top_k=1)
        assert results[0].emoji == self.index.metadata[0]["emoji"]
        assert results[0].score == pytest.approx(1.0, abs=1e-6)

    def test_lookup_found(self):
        """lookup returns metadata for a known emoji."""
        first_emoji = self.index.metadata[0]["emoji"]
        entry = self.index.lookup(first_emoji)
        assert entry is not None
        assert entry["emoji"] == first_emoji
        assert "short_name" in entry

    def test_lookup_not_found(self):
        """lookup returns None for an unknown emoji."""
        assert self.index.lookup("🤖") is None

    def test_search_zero_vector(self):
        """Searching with a zero vector returns empty list."""
        results = self.index.search(np.zeros(1536), top_k=5)
        assert results == []
