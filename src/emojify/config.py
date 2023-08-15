"""Configuration and constants for emojify."""

import os
import sys
from pathlib import Path

EMBEDDING_MODEL = "text-embedding-ada-002"
DECODE_MODEL = "gpt-3.5-turbo"

# Data paths — relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_PATH = PROCESSED_DIR / "emoji_metadata.json"
INDEX_PATH = PROCESSED_DIR / "emoji_index.npz"
CACHE_PATH = PROJECT_ROOT / "cache.sqlite"


def get_openai_api_key() -> str:
    """Return the OpenAI API key from the environment, or exit with a clear error."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Set it with: export OPENAI_API_KEY='your-key-here'",
            file=sys.stderr,
        )
        sys.exit(1)
    return key
