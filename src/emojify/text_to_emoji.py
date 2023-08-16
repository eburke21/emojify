"""Text-to-emoji pipeline — embed a query and find the closest emoji."""

from emojify.embeddings import get_embedding
from emojify.index import EmojiIndex, EmojiMatch


def text_to_emoji(
    query: str,
    index: EmojiIndex,
    top_k: int = 5,
    diverse: bool = False,
    target_count: int = 3,
) -> list[EmojiMatch]:
    """Embed the query text and return the top-K most similar emoji.

    Args:
        query: Natural language text to match against emoji.
        index: Preloaded EmojiIndex with embeddings and metadata.
        top_k: Number of raw results to retrieve before diversity filtering.
        diverse: If True, apply category-based diversity filtering.
        target_count: Number of results to return after diversity filtering
                      (only used when diverse=True).

    Returns:
        List of EmojiMatch results sorted by descending similarity score.
    """
    query_embedding = get_embedding(query)
    results = index.search(query_embedding, top_k=top_k)

    if diverse:
        from emojify.diversity import apply_diversity_filter

        results = apply_diversity_filter(
            matches=results,
            metadata=index.metadata,
            emoji_to_idx=index.emoji_to_idx,
            target_count=target_count,
        )

    return results


def suggest(query: str, index: EmojiIndex, count: int = 5) -> str:
    """Return a string of emoji characters for a query — no scores or metadata.

    This is the quick copy-paste mode: embed the query, diversity-filter,
    and return just the emoji characters concatenated.
    """
    matches = text_to_emoji(
        query,
        index,
        top_k=count * 3,  # fetch extra so diversity filter has room
        diverse=True,
        target_count=count,
    )
    return "".join(m.emoji for m in matches)
