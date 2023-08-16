"""Parse CLDR and Emojilib sources, merge into unified emoji metadata."""

import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"

# Regex to detect emoji characters (broad match for Unicode emoji ranges)
EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F"  # Emoticons
    r"\U0001F300-\U0001F5FF"   # Misc Symbols and Pictographs
    r"\U0001F680-\U0001F6FF"   # Transport and Map
    r"\U0001F1E0-\U0001F1FF"   # Flags
    r"\U0001F900-\U0001F9FF"   # Supplemental Symbols
    r"\U0001FA00-\U0001FA6F"   # Chess Symbols
    r"\U0001FA70-\U0001FAFF"   # Symbols Extended-A
    r"\U00002702-\U000027B0"   # Dingbats
    r"\U0000FE00-\U0000FE0F"   # Variation Selectors
    r"\U0000200D"              # ZWJ
    r"\U000024C2-\U0001F251"   # Enclosed chars
    r"\U00002600-\U000026FF"   # Misc symbols
    r"\U00002300-\U000023FF"   # Misc Technical
    r"\U00002B50-\U00002B55"   # Stars
    r"\U000023E9-\U000023F3"   # Media symbols
    r"\U000023F8-\U000023FA"   # Media symbols 2
    r"\U0000231A-\U0000231B"   # Watch/Hourglass
    r"\U00003030\U000025AA-\U000025AB\U000025FB-\U000025FE"
    r"\U00002934-\U00002935"
    r"\U000025B6\U000025C0"
    r"\U00002194-\U00002199"
    r"\U000021A9-\U000021AA"
    r"]"
)

# Unicode general categories for emoji grouping
CATEGORY_MAP = {
    "Smileys & Emotion": ["face", "smiley", "emotion", "laugh", "smile", "cry", "angry", "love", "heart"],
    "People & Body": ["person", "hand", "gesture", "body", "family", "baby", "man", "woman", "boy", "girl"],
    "Animals & Nature": ["animal", "nature", "plant", "flower", "tree", "bird", "dog", "cat", "fish", "bug"],
    "Food & Drink": ["food", "drink", "fruit", "vegetable", "meal", "pizza", "coffee", "beer", "wine", "cake"],
    "Travel & Places": ["travel", "place", "building", "vehicle", "car", "train", "plane", "ship", "house", "city"],
    "Activities": ["sport", "game", "ball", "trophy", "medal", "music", "art", "theater", "dice"],
    "Objects": ["object", "tool", "phone", "computer", "book", "clock", "light", "key", "money", "mail"],
    "Symbols": ["symbol", "sign", "arrow", "number", "letter", "zodiac", "warning", "check", "cross", "star"],
    "Flags": ["flag"],
}


@dataclass
class EmojiRecord:
    emoji: str
    short_name: str
    keywords: list[str] = field(default_factory=list)
    category: str = "Symbols"
    description: str = ""


def is_emoji(char: str) -> bool:
    """Check if a string looks like an emoji (not punctuation/letters)."""
    return bool(EMOJI_RE.search(char)) or (len(char) > 1 and any(ord(c) > 0x2000 for c in char))


def parse_cldr(path: Path) -> dict[str, EmojiRecord]:
    """Parse Unicode CLDR en.xml into emoji records."""
    tree = ET.parse(path)
    root = tree.getroot()

    # First pass: collect keywords and tts names
    keywords: dict[str, list[str]] = {}
    names: dict[str, str] = {}

    for annotation in root.iter("annotation"):
        cp = annotation.get("cp", "")
        if not cp:
            continue

        if annotation.get("type") == "tts":
            names[cp] = annotation.text.strip() if annotation.text else ""
        else:
            kws = [kw.strip().lower() for kw in (annotation.text or "").split("|")]
            keywords[cp] = [kw for kw in kws if kw]

    # Build records for actual emoji only
    records: dict[str, EmojiRecord] = {}
    for cp in set(list(keywords.keys()) + list(names.keys())):
        if not is_emoji(cp):
            continue
        name = names.get(cp, "")
        kws = keywords.get(cp, [])
        if name or kws:
            records[cp] = EmojiRecord(emoji=cp, short_name=name, keywords=kws)

    return records


def parse_emojilib(path: Path) -> dict[str, list[str]]:
    """Parse Emojilib JSON into emoji -> keyword list mapping."""
    with open(path) as f:
        data = json.load(f)

    result: dict[str, list[str]] = {}
    for emoji, kws in data.items():
        cleaned = []
        seen = set()
        for kw in kws:
            kw_clean = kw.strip().lower().replace("_", " ")
            if kw_clean and kw_clean not in seen:
                cleaned.append(kw_clean)
                seen.add(kw_clean)
        result[emoji] = cleaned

    return result


def guess_category(short_name: str, keywords: list[str]) -> str:
    """Guess the Unicode general category based on name and keywords."""
    all_text = (short_name + " " + " ".join(keywords)).lower()
    for category, indicators in CATEGORY_MAP.items():
        if any(ind in all_text for ind in indicators):
            return category
    if "flag" in all_text:
        return "Flags"
    return "Symbols"


def merge_sources(cldr: dict[str, EmojiRecord], emojilib: dict[str, list[str]]) -> list[EmojiRecord]:
    """Merge CLDR and Emojilib into a unified dataset."""
    all_emoji = set(list(cldr.keys()) + list(emojilib.keys()))
    records: list[EmojiRecord] = []

    for emoji in sorted(all_emoji):
        cldr_entry = cldr.get(emoji)
        emojilib_kws = emojilib.get(emoji, [])

        if cldr_entry:
            # Start with CLDR data
            record = EmojiRecord(
                emoji=emoji,
                short_name=cldr_entry.short_name,
                keywords=list(cldr_entry.keywords),
            )
            # Add Emojilib keywords that aren't already present
            existing = {kw.lower() for kw in record.keywords}
            for kw in emojilib_kws:
                if kw.lower() not in existing:
                    record.keywords.append(kw)
                    existing.add(kw.lower())
        elif emojilib_kws:
            # Emojilib only — use first keyword as name
            name = emojilib_kws[0] if emojilib_kws else emoji
            record = EmojiRecord(
                emoji=emoji,
                short_name=name,
                keywords=emojilib_kws,
            )
        else:
            continue

        # Filter: must have at least a name and some keywords
        if not record.short_name and not record.keywords:
            continue

        # Guess category
        record.category = guess_category(record.short_name, record.keywords)
        records.append(record)

    return records


def main() -> None:
    cldr_path = RAW_DIR / "en.xml"
    emojilib_path = RAW_DIR / "emojilib.json"

    if not cldr_path.exists() or not emojilib_path.exists():
        print("Error: Raw data files missing. Run fetch_data.py first.", file=sys.stderr)
        sys.exit(1)

    print("Parsing CLDR annotations...")
    cldr = parse_cldr(cldr_path)
    print(f"  Found {len(cldr)} emoji in CLDR")

    print("Parsing Emojilib...")
    emojilib = parse_emojilib(emojilib_path)
    print(f"  Found {len(emojilib)} emoji in Emojilib")

    print("Merging sources...")
    records = merge_sources(cldr, emojilib)
    print(f"  Merged dataset: {len(records)} emoji")

    # Save intermediate result (without descriptions yet)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "emoji_metadata.json"
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
