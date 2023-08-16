"""Tests for the diversity filter."""

import pytest

from emojify.index import EmojiMatch
from emojify.diversity import apply_diversity_filter


def _make_match(emoji: str, score: float) -> EmojiMatch:
    """Helper to build a test EmojiMatch."""
    return EmojiMatch(emoji=emoji, score=score, short_name=f"name_{emoji}", keywords=[])


def _make_metadata_and_idx(entries: list[tuple[str, str]]):
    """Build test metadata and emoji_to_idx from (emoji, category) pairs.

    Returns (metadata_list, emoji_to_idx_dict).
    """
    metadata = [
        {
            "emoji": emoji,
            "short_name": f"name_{emoji}",
            "keywords": [],
            "category": cat,
            "description": f"desc {emoji}",
        }
        for emoji, cat in entries
    ]
    emoji_to_idx = {emoji: i for i, (emoji, _) in enumerate(entries)}
    return metadata, emoji_to_idx


class TestDiversityFilter:
    """Tests for apply_diversity_filter."""

    def test_filters_same_category(self):
        """Five emoji from one category → only 1 kept, rest filled from others."""
        entries = [
            ("😀", "Smileys & Emotion"),
            ("😃", "Smileys & Emotion"),
            ("😄", "Smileys & Emotion"),
            ("😁", "Smileys & Emotion"),
            ("😊", "Smileys & Emotion"),
            ("🍕", "Food & Drink"),
            ("🚀", "Travel & Places"),
        ]
        metadata, emoji_to_idx = _make_metadata_and_idx(entries)

        matches = [
            _make_match("😀", 0.95),
            _make_match("😃", 0.93),
            _make_match("😄", 0.91),
            _make_match("😁", 0.89),
            _make_match("😊", 0.87),
            _make_match("🍕", 0.80),
            _make_match("🚀", 0.75),
        ]

        result = apply_diversity_filter(
            matches, metadata, emoji_to_idx,
            max_per_category=1, target_count=3,
        )

        assert len(result) == 3
        emojis = [m.emoji for m in result]
        # Only one smiley should remain (the highest-scoring one)
        smiley_count = sum(1 for e in emojis if e in {"😀", "😃", "😄", "😁", "😊"})
        assert smiley_count == 1
        assert "😀" in emojis  # highest-scoring smiley
        # Other slots filled from different categories
        assert "🍕" in emojis or "🚀" in emojis

    def test_preserves_order(self):
        """Output maintains descending score order within the filtered set."""
        entries = [
            ("😀", "Smileys & Emotion"),
            ("🍕", "Food & Drink"),
            ("🚀", "Travel & Places"),
            ("⚽", "Activities"),
        ]
        metadata, emoji_to_idx = _make_metadata_and_idx(entries)

        matches = [
            _make_match("😀", 0.95),
            _make_match("🍕", 0.90),
            _make_match("🚀", 0.85),
            _make_match("⚽", 0.80),
        ]

        result = apply_diversity_filter(
            matches, metadata, emoji_to_idx,
            max_per_category=1, target_count=3,
        )

        scores = [m.score for m in result]
        assert scores == sorted(scores, reverse=True)

    def test_relaxes_when_not_enough_categories(self):
        """If all matches are same category, relaxes to fill target_count."""
        entries = [
            ("😀", "Smileys & Emotion"),
            ("😃", "Smileys & Emotion"),
            ("😄", "Smileys & Emotion"),
        ]
        metadata, emoji_to_idx = _make_metadata_and_idx(entries)

        matches = [
            _make_match("😀", 0.95),
            _make_match("😃", 0.93),
            _make_match("😄", 0.91),
        ]

        result = apply_diversity_filter(
            matches, metadata, emoji_to_idx,
            max_per_category=1, target_count=3,
        )

        # Pass 1 gets 1, pass 2 fills the remaining 2
        assert len(result) == 3

    def test_empty_input(self):
        """Empty matches list returns empty result."""
        result = apply_diversity_filter([], [], {}, target_count=3)
        assert result == []

    def test_target_count_larger_than_matches(self):
        """If target_count > len(matches), return all available matches."""
        entries = [("😀", "Smileys & Emotion"), ("🍕", "Food & Drink")]
        metadata, emoji_to_idx = _make_metadata_and_idx(entries)

        matches = [_make_match("😀", 0.95), _make_match("🍕", 0.90)]

        result = apply_diversity_filter(
            matches, metadata, emoji_to_idx,
            max_per_category=1, target_count=5,
        )

        assert len(result) == 2  # Only 2 available

    def test_max_per_category_two(self):
        """max_per_category=2 allows two emoji from the same category."""
        entries = [
            ("😀", "Smileys & Emotion"),
            ("😃", "Smileys & Emotion"),
            ("😄", "Smileys & Emotion"),
            ("🍕", "Food & Drink"),
        ]
        metadata, emoji_to_idx = _make_metadata_and_idx(entries)

        matches = [
            _make_match("😀", 0.95),
            _make_match("😃", 0.93),
            _make_match("😄", 0.91),
            _make_match("🍕", 0.80),
        ]

        result = apply_diversity_filter(
            matches, metadata, emoji_to_idx,
            max_per_category=2, target_count=3,
        )

        assert len(result) == 3
        emojis = [m.emoji for m in result]
        smiley_count = sum(1 for e in emojis if e in {"😀", "😃", "😄"})
        assert smiley_count == 2
