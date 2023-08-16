"""Embedding API calls, batching, and SQLite caching."""

import sqlite3
import time
from pathlib import Path

import numpy as np
import openai
from rich.progress import Progress

from emojify.config import EMBEDDING_MODEL, CACHE_PATH, get_openai_api_key

EMBEDDING_DIM = 1536


def _ensure_api_key() -> None:
    """Set the OpenAI API key if not already configured."""
    if not openai.api_key:
        openai.api_key = get_openai_api_key()


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """SQLite-backed cache for query embeddings."""

    def __init__(self, db_path: Path = CACHE_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embedding_cache "
                "(text TEXT PRIMARY KEY, embedding BLOB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
        return self._conn

    def get(self, text: str) -> np.ndarray | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text = ?", (text,)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float64).copy()

    def put(self, text: str, embedding: np.ndarray) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text, embedding) VALUES (?, ?)",
            (text, embedding.astype(np.float64).tobytes()),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# Module-level cache instance (can be disabled by setting to None)
_cache: EmbeddingCache | None = EmbeddingCache()


def disable_cache() -> None:
    """Disable the embedding cache."""
    global _cache
    if _cache is not None:
        _cache.close()
    _cache = None


def enable_cache(db_path: Path = CACHE_PATH) -> None:
    """Enable or re-enable the embedding cache."""
    global _cache
    _cache = EmbeddingCache(db_path)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def _call_embedding_api(text: str) -> np.ndarray:
    """Call OpenAI embedding API for a single text. No caching."""
    _ensure_api_key()
    response = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
    return np.array(response["data"][0]["embedding"], dtype=np.float64)


def _call_embedding_api_batch(texts: list[str]) -> np.ndarray:
    """Call OpenAI embedding API for a batch of texts. No caching."""
    _ensure_api_key()
    response = openai.Embedding.create(input=texts, model=EMBEDDING_MODEL)
    # Response data may not be in order — sort by index
    sorted_data = sorted(response["data"], key=lambda x: x["index"])
    return np.array([d["embedding"] for d in sorted_data], dtype=np.float64)


def get_embedding(text: str, use_cache: bool = True) -> np.ndarray:
    """Get embedding for a single text, with optional cache lookup."""
    if use_cache and _cache is not None:
        cached = _cache.get(text)
        if cached is not None:
            return cached

    embedding = _call_embedding_api(text)

    if use_cache and _cache is not None:
        _cache.put(text, embedding)

    return embedding


def get_embeddings_batch(
    texts: list[str],
    batch_size: int = 100,
    max_retries: int = 3,
    show_progress: bool = True,
) -> np.ndarray:
    """Batch-embed texts with rate limiting, retries, and progress bar.

    Returns array of shape (len(texts), 1536).
    """
    _ensure_api_key()
    all_embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float64)

    batches = [
        texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
    ]

    progress_ctx = Progress() if show_progress else None

    if progress_ctx:
        progress_ctx.start()
        task = progress_ctx.add_task("Embedding...", total=len(texts))

    try:
        offset = 0
        for batch in batches:
            for attempt in range(max_retries):
                try:
                    batch_result = _call_embedding_api_batch(batch)
                    all_embeddings[offset : offset + len(batch)] = batch_result
                    break
                except openai.error.RateLimitError:
                    wait = 2 ** (attempt + 1)
                    if progress_ctx:
                        progress_ctx.console.print(
                            f"[yellow]Rate limited. Waiting {wait}s...[/yellow]"
                        )
                    time.sleep(wait)
                except openai.error.APIError as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise e

            if progress_ctx:
                progress_ctx.update(task, advance=len(batch))
            offset += len(batch)
    finally:
        if progress_ctx:
            progress_ctx.stop()

    return all_embeddings
