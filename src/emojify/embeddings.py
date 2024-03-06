"""Embedding API calls, batching, and SQLite caching."""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from pathlib import Path

import numpy as np

from emojify.config import EMBEDDING_MODEL, CACHE_PATH, get_openai_api_key

EMBEDDING_DIM = 1536
API_TIMEOUT = 15  # seconds

_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


def _make_client() -> OpenAI:
    """Create a new OpenAI client (used for batch calls only)."""
    return OpenAI(
        api_key=get_openai_api_key(),
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


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
    """Call OpenAI embedding API for a single text using urllib.

    Bypasses httpx/openai SDK to avoid hanging on some systems.
    Uses a reliable socket-level timeout via urllib.
    """
    api_key = get_openai_api_key()
    payload = json.dumps({"input": text, "model": EMBEDDING_MODEL}).encode()
    req = urllib.request.Request(
        _OPENAI_EMBED_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"OpenAI API error {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach OpenAI API: {e.reason}") from e

    return np.array(body["data"][0]["embedding"], dtype=np.float64)


def _call_embedding_api_batch(texts: list[str]) -> np.ndarray:
    """Call OpenAI embedding API for a batch of texts. No caching."""
    client = _make_client()
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return np.array([d.embedding for d in sorted_data], dtype=np.float64)


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
                except RateLimitError:
                    wait = 2 ** (attempt + 1)
                    if progress_ctx:
                        progress_ctx.console.print(
                            f"[yellow]Rate limited. Waiting {wait}s...[/yellow]"
                        )
                    time.sleep(wait)
                except APIError as e:
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
