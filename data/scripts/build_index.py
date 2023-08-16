"""Embed all emoji descriptions via ada-002 and save to emoji_index.npz."""

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path so we can import emojify
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from emojify.embeddings import get_embeddings_batch, disable_cache

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"


def main() -> None:
    metadata_path = PROCESSED_DIR / "emoji_metadata.json"
    index_path = PROCESSED_DIR / "emoji_index.npz"

    if not metadata_path.exists():
        print("Error: emoji_metadata.json not found. Run merge_sources.py first.", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} emoji records.")

    descriptions = [r["description"] for r in records]
    emoji_list = [r["emoji"] for r in records]

    # Don't cache index-building embeddings (they're stored in the .npz)
    disable_cache()

    print(f"Embedding {len(descriptions)} descriptions via ada-002...")
    embeddings = get_embeddings_batch(descriptions, batch_size=100)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        index_path,
        embeddings=embeddings,
        emoji_list=np.array(emoji_list, dtype=object),
    )
    print(f"  Saved to {index_path}")

    # Verify
    loaded = np.load(index_path, allow_pickle=True)
    assert loaded["embeddings"].shape == embeddings.shape, "Shape mismatch after reload!"
    assert len(loaded["emoji_list"]) == len(records), "Count mismatch after reload!"
    print(f"  Verification passed: {loaded['embeddings'].shape[0]} embeddings, {loaded['embeddings'].shape[1]} dimensions")


if __name__ == "__main__":
    main()
