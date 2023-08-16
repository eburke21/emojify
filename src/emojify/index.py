"""EmojiIndex class — loads precomputed embeddings and searches by cosine similarity."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emojify.config import METADATA_PATH, INDEX_PATH


@dataclass
class EmojiMatch:
    """A single search result."""
    emoji: str
    score: float
    short_name: str
    keywords: list[str]


class EmojiIndex:
    """Precomputed emoji embedding index with cosine similarity search."""

    def __init__(
        self,
        metadata_path: Path = METADATA_PATH,
        index_path: Path = INDEX_PATH,
    ):
        with open(metadata_path) as f:
            self.metadata: list[dict] = json.load(f)

        data = np.load(index_path, allow_pickle=True)
        self.embeddings: np.ndarray = data["embeddings"].astype(np.float64)
        emoji_list = data["emoji_list"]

        # Verify alignment
        assert len(self.metadata) == self.embeddings.shape[0], (
            f"Metadata ({len(self.metadata)}) and index ({self.embeddings.shape[0]}) "
            "have different counts. Rebuild the index."
        )

        # Pre-normalize embeddings for faster cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        self._normed = self.embeddings / norms

        # Reverse lookup: emoji character -> index
        self.emoji_to_idx: dict[str, int] = {
            self.metadata[i]["emoji"]: i for i in range(len(self.metadata))
        }

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[EmojiMatch]:
        """Return top-K emoji by cosine similarity to the query embedding."""
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        normed_query = query_embedding / query_norm

        similarities = self._normed @ normed_query
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            EmojiMatch(
                emoji=self.metadata[i]["emoji"],
                score=float(similarities[i]),
                short_name=self.metadata[i]["short_name"],
                keywords=self.metadata[i]["keywords"],
            )
            for i in top_indices
        ]

    def lookup(self, emoji: str) -> dict | None:
        """Return the metadata entry for a given emoji character."""
        idx = self.emoji_to_idx.get(emoji)
        if idx is None:
            return None
        return self.metadata[idx]
