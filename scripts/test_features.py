"""Integration sanity check — test both pipelines against real data.

Requires:
  - OPENAI_API_KEY set in environment
  - emoji_index.npz built (run `make build-index` first)

Usage:
  python scripts/test_features.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emojify.index import EmojiIndex
from emojify.text_to_emoji import text_to_emoji, suggest
from emojify.decoder import decode_emoji


def test_text_to_emoji(index: EmojiIndex) -> None:
    """Test text_to_emoji with a few known queries."""
    print("=" * 60)
    print("TEXT → EMOJI TESTS")
    print("=" * 60)

    queries = [
        ("happy birthday", ["🎂", "🎉", "🥳", "🎁", "🎈"]),
        ("pizza for dinner", ["🍕"]),
        ("rocket launch", ["🚀"]),
        ("sad and crying", ["😢", "😭", "😿", "💧"]),
        ("just deployed to production at 2am", ["🚀", "💻", "🌙", "😴"]),
    ]

    for query, expected in queries:
        results = text_to_emoji(query, index, top_k=5)
        result_emoji = [r.emoji for r in results]
        hit = any(e in result_emoji for e in expected)

        status = "✅" if hit else "❌"
        print(f"\n{status} Query: \"{query}\"")
        for r in results:
            marker = " ← expected" if r.emoji in expected else ""
            print(f"   {r.emoji}  {r.score:.3f}  {r.short_name}{marker}")


def test_suggest(index: EmojiIndex) -> None:
    """Test the suggest helper."""
    print("\n" + "=" * 60)
    print("SUGGEST MODE TESTS")
    print("=" * 60)

    queries = [
        "just shipped to production",
        "good morning",
        "happy birthday",
        "it's freezing outside",
    ]

    for query in queries:
        result = suggest(query, index, count=5)
        print(f"\n  \"{query}\"")
        print(f"   → {result}")


def test_decode(index: EmojiIndex) -> None:
    """Test the emoji decoder."""
    print("\n" + "=" * 60)
    print("EMOJI → TEXT TESTS")
    print("=" * 60)

    test_cases = [
        "🍕🍺📺",
        "🎉🎂🎁",
        "🚀🔥💯",
        "😭💔",
    ]

    for emoji_str in test_cases:
        result = decode_emoji(emoji_str, index, use_llm=True)

        print(f"\n  Input: {emoji_str}")
        print("  Individual meanings:")
        for desc in result.individual:
            status = "" if desc.found_in_index else " [UNKNOWN]"
            print(f"    {desc.emoji}  {desc.short_name} — {', '.join(desc.keywords[:5])}{status}")
        print(f"  Combined: \"{result.combined_interpretation}\"")


def main() -> None:
    print("Loading emoji index...")
    try:
        index = EmojiIndex()
    except FileNotFoundError:
        print(
            "Error: emoji_index.npz or emoji_metadata.json not found.\n"
            "Run `make build-index` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(index.metadata)} emoji.\n")

    test_text_to_emoji(index)
    test_suggest(index)
    test_decode(index)

    print("\n" + "=" * 60)
    print("DONE — review results above for quality.")
    print("=" * 60)


if __name__ == "__main__":
    main()
