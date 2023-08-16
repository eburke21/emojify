"""Diversity filtering — category-based deduplication for emoji sequences.

Without diversity filtering, a query like "I'm so happy" returns five nearly
identical smiling faces (😀😃😄😁😊).  The filter keeps at most
`max_per_category` emoji from each Unicode category, filling the remaining
slots with the next highest-scoring emoji from under-represented categories.
"""

from emojify.index import EmojiMatch


def apply_diversity_filter(
    matches: list[EmojiMatch],
    metadata: list[dict],
    emoji_to_idx: dict[str, int],
    max_per_category: int = 1,
    target_count: int = 3,
) -> list[EmojiMatch]:
    """Select a diverse subset of emoji matches across Unicode categories.

    Iterates through `matches` (already sorted by descending similarity score).
    For each match, checks its Unicode category from `metadata`.  If that
    category already has `max_per_category` emoji in the output, it is skipped.
    Stops when `target_count` results are collected.

    If there aren't enough distinct categories to fill `target_count`, the
    constraint is relaxed: a second pass allows additional emoji from the most
    relevant categories.

    Args:
        matches: Ranked EmojiMatch list (descending by score).
        metadata: Full metadata list (same order as the index).
        emoji_to_idx: Mapping from emoji character to metadata index.
        max_per_category: Max emoji allowed per category before skipping.
        target_count: Desired number of results in the output.

    Returns:
        A diversity-filtered list of EmojiMatch, still sorted by score.
    """
    if not matches:
        return []

    if target_count <= 0:
        return []

    # --- Pass 1: strict category limit ---
    selected: list[EmojiMatch] = []
    category_counts: dict[str, int] = {}

    for match in matches:
        if len(selected) >= target_count:
            break

        idx = emoji_to_idx.get(match.emoji)
        category = metadata[idx]["category"] if idx is not None else "Unknown"

        if category_counts.get(category, 0) < max_per_category:
            selected.append(match)
            category_counts[category] = category_counts.get(category, 0) + 1

    # --- Pass 2: relax constraint if not enough results ---
    if len(selected) < target_count:
        already = {m.emoji for m in selected}
        for match in matches:
            if len(selected) >= target_count:
                break
            if match.emoji not in already:
                selected.append(match)

    return selected
