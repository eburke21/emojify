"""Generate natural language descriptions for each emoji and update metadata."""

import json
import sys
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"


def generate_description(emoji: str, short_name: str, keywords: list[str]) -> str:
    """Generate a template-based natural language description for an emoji."""
    keyword_str = ", ".join(keywords[:8])
    return f"{emoji} ({short_name}) — commonly used to express: {keyword_str}"


def main() -> None:
    metadata_path = PROCESSED_DIR / "emoji_metadata.json"
    if not metadata_path.exists():
        print("Error: emoji_metadata.json not found. Run merge_sources.py first.", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path) as f:
        records = json.load(f)

    print(f"Generating descriptions for {len(records)} emoji...")
    for record in records:
        record["description"] = generate_description(
            record["emoji"],
            record["short_name"],
            record["keywords"],
        )

    with open(metadata_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Verify
    sample = records[0]
    print(f"  Sample: {sample['description'][:80]}...")
    lengths = [len(r["description"]) for r in records]
    print(f"  Description lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
    print(f"  Saved {len(records)} entries to {metadata_path}")


if __name__ == "__main__":
    main()
