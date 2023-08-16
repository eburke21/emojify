"""Manual end-to-end search verification.

Loads the real index, embeds test queries, and prints top results.
Requires: OPENAI_API_KEY set, emoji_index.npz built.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emojify.index import EmojiIndex
from emojify.embeddings import get_embedding

QUERIES = [
    "I'm so happy",
    "pizza for dinner",
    "rocket launch",
    "sad and crying",
    "just deployed to production at 2am",
    "happy birthday",
    "it's freezing outside",
    "stuck in traffic",
]


def main() -> None:
    print("Loading index...")
    index = EmojiIndex()
    print(f"  {len(index.metadata)} emoji, {index.embeddings.shape[1]}-dim embeddings\n")

    for query in QUERIES:
        embedding = get_embedding(query)
        results = index.search(embedding, top_k=5)

        print(f'Query: "{query}"')
        for r in results:
            kw_str = ", ".join(r.keywords[:4])
            print(f"  {r.emoji} {r.score:.3f}  {r.short_name} — {kw_str}")
        print()


if __name__ == "__main__":
    main()
