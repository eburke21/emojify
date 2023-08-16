"""Tests for the data pipeline: CLDR parsing, Emojilib parsing, merging, descriptions."""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "scripts"))

from merge_sources import parse_cldr, parse_emojilib, merge_sources
from generate_descriptions import generate_description

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def test_parse_cldr():
    """CLDR parser returns ~1600+ entries with expected structure."""
    cldr = parse_cldr(RAW_DIR / "en.xml")
    assert len(cldr) >= 1500, f"Expected >=1500 CLDR entries, got {len(cldr)}"

    # Spot-check: grinning face
    assert "😀" in cldr
    assert cldr["😀"].short_name == "grinning face"
    assert "grin" in cldr["😀"].keywords or "grinning" in cldr["😀"].keywords

    # Spot-check: surfing
    assert "🏄" in cldr
    assert "surfing" in cldr["🏄"].keywords or "surf" in cldr["🏄"].keywords


def test_parse_emojilib():
    """Emojilib parser returns ~1900 entries with normalized keywords."""
    emojilib = parse_emojilib(RAW_DIR / "emojilib.json")
    assert len(emojilib) >= 1800, f"Expected >=1800 Emojilib entries, got {len(emojilib)}"

    # Spot-check: fire emoji
    assert "🔥" in emojilib
    kws = emojilib["🔥"]
    assert "fire" in kws
    assert "hot" in kws

    # Keywords should be lowercase
    for emoji, keywords in list(emojilib.items())[:50]:
        for kw in keywords:
            assert kw == kw.lower(), f"Keyword '{kw}' for {emoji} not lowercase"


def test_merge():
    """Merged dataset has >=1500 entries, no duplicates, all have >=2 keywords."""
    cldr = parse_cldr(RAW_DIR / "en.xml")
    emojilib = parse_emojilib(RAW_DIR / "emojilib.json")
    merged = merge_sources(cldr, emojilib)

    assert len(merged) >= 1500, f"Expected >=1500 merged entries, got {len(merged)}"

    # No duplicate emoji characters
    emoji_chars = [r.emoji for r in merged]
    assert len(emoji_chars) == len(set(emoji_chars)), "Duplicate emoji found in merged data"

    # Most entries should have at least 2 keywords
    few_keywords = [r for r in merged if len(r.keywords) < 2]
    assert len(few_keywords) < 20, f"Too many entries with <2 keywords: {len(few_keywords)}"


def test_merge_combines_keywords():
    """Merging adds Emojilib keywords not present in CLDR."""
    cldr = parse_cldr(RAW_DIR / "en.xml")
    emojilib = parse_emojilib(RAW_DIR / "emojilib.json")
    merged = merge_sources(cldr, emojilib)

    # Find pizza — CLDR has basic keywords, Emojilib should add more
    pizza = next((r for r in merged if r.emoji == "🍕"), None)
    assert pizza is not None
    assert "pizza" in pizza.keywords
    assert len(pizza.keywords) >= 5, "Expected merged keywords from both sources"


def test_descriptions():
    """Every entry in emoji_metadata.json has a valid description."""
    metadata_path = PROCESSED_DIR / "emoji_metadata.json"
    assert metadata_path.exists(), "emoji_metadata.json not found — run the pipeline first"

    with open(metadata_path) as f:
        records = json.load(f)

    assert len(records) >= 1500

    for record in records:
        desc = record["description"]
        assert desc, f"Empty description for {record['emoji']}"
        assert len(desc) >= 30, f"Description too short for {record['emoji']}: {desc}"
        assert len(desc) <= 300, f"Description too long for {record['emoji']}: {desc}"
        assert "commonly used to express:" in desc, f"Missing template format for {record['emoji']}"


def test_generate_description_format():
    """generate_description returns the expected template format."""
    desc = generate_description("🍕", "pizza", ["cheese", "food", "slice"])
    assert desc == "🍕 (pizza) — commonly used to express: cheese, food, slice"
