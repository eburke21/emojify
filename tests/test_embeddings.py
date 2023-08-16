"""Tests for the embedding API wrapper and SQLite cache."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from emojify.embeddings import (
    get_embedding,
    get_embeddings_batch,
    EmbeddingCache,
    EMBEDDING_DIM,
    _call_embedding_api,
    _call_embedding_api_batch,
)


def _make_fake_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    """Return a deterministic fake embedding vector."""
    rng = np.random.RandomState(42)
    return rng.randn(dim).tolist()


def _mock_single_response(embedding: list[float]) -> dict:
    """Build a fake OpenAI Embedding.create response for a single text."""
    return {"data": [{"embedding": embedding, "index": 0}]}


def _mock_batch_response(embeddings: list[list[float]]) -> dict:
    """Build a fake OpenAI Embedding.create response for a batch."""
    return {
        "data": [
            {"embedding": emb, "index": i}
            for i, emb in enumerate(embeddings)
        ]
    }


@patch("emojify.embeddings.openai.Embedding.create")
@patch("emojify.embeddings._ensure_api_key")
def test_single_embedding(mock_key, mock_create):
    """get_embedding returns a numpy array of shape (1536,)."""
    fake = _make_fake_embedding()
    mock_create.return_value = _mock_single_response(fake)

    from emojify.embeddings import disable_cache
    disable_cache()

    result = _call_embedding_api("hello world")
    assert isinstance(result, np.ndarray)
    assert result.shape == (EMBEDDING_DIM,)
    np.testing.assert_array_almost_equal(result, np.array(fake))

    from emojify.embeddings import enable_cache
    enable_cache()


@patch("emojify.embeddings.openai.Embedding.create")
@patch("emojify.embeddings._ensure_api_key")
def test_batch_embedding(mock_key, mock_create):
    """get_embeddings_batch returns shape (N, 1536) for N texts."""
    rng = np.random.RandomState(123)
    n = 5
    fakes = [rng.randn(EMBEDDING_DIM).tolist() for _ in range(n)]
    mock_create.return_value = _mock_batch_response(fakes)

    texts = [f"text {i}" for i in range(n)]
    result = get_embeddings_batch(texts, batch_size=10, show_progress=False)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n, EMBEDDING_DIM)
    for i in range(n):
        np.testing.assert_array_almost_equal(result[i], np.array(fakes[i]))


class TestEmbeddingCache:
    """Tests for the SQLite embedding cache."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_cache.sqlite"
        self.cache = EmbeddingCache(db_path=self.db_path)

    def teardown_method(self):
        self.cache.close()

    def test_cache_miss_returns_none(self):
        assert self.cache.get("nonexistent") is None

    def test_cache_roundtrip(self):
        """Store and retrieve an embedding."""
        vec = np.random.randn(EMBEDDING_DIM)
        self.cache.put("hello", vec)
        result = self.cache.get("hello")
        assert result is not None
        np.testing.assert_array_almost_equal(result, vec)

    def test_cache_overwrite(self):
        """Putting the same key twice overwrites."""
        vec1 = np.ones(EMBEDDING_DIM)
        vec2 = np.ones(EMBEDDING_DIM) * 2
        self.cache.put("key", vec1)
        self.cache.put("key", vec2)
        result = self.cache.get("key")
        np.testing.assert_array_almost_equal(result, vec2)


@patch("emojify.embeddings._call_embedding_api")
def test_cache_hit(mock_api):
    """get_embedding with cache: API called once for repeated queries."""
    fake = np.random.randn(EMBEDDING_DIM)
    mock_api.return_value = fake.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        from emojify.embeddings import enable_cache, disable_cache
        enable_cache(Path(tmpdir) / "cache.sqlite")

        result1 = get_embedding("same query", use_cache=True)
        result2 = get_embedding("same query", use_cache=True)

        assert mock_api.call_count == 1
        np.testing.assert_array_almost_equal(result1, result2)

        disable_cache()
        enable_cache()


@patch("emojify.embeddings._call_embedding_api")
def test_cache_miss_different_texts(mock_api):
    """get_embedding with cache: API called once per unique query."""
    fake1 = np.random.randn(EMBEDDING_DIM)
    fake2 = np.random.randn(EMBEDDING_DIM)
    mock_api.side_effect = [fake1.copy(), fake2.copy()]

    with tempfile.TemporaryDirectory() as tmpdir:
        from emojify.embeddings import enable_cache, disable_cache
        enable_cache(Path(tmpdir) / "cache.sqlite")

        get_embedding("query one", use_cache=True)
        get_embedding("query two", use_cache=True)

        assert mock_api.call_count == 2

        disable_cache()
        enable_cache()
