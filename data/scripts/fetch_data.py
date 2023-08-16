"""Download raw emoji data sources: Unicode CLDR annotations and Emojilib."""

import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

SOURCES = {
    "en.xml": "https://raw.githubusercontent.com/unicode-org/cldr/main/common/annotations/en.xml",
    "emojilib.json": "https://raw.githubusercontent.com/muan/emojilib/main/dist/emoji-en-US.json",
}


def download(url: str, dest: Path, retries: int = 1) -> None:
    """Download a URL to a local file with retry on failure."""
    for attempt in range(retries + 1):
        try:
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            size = dest.stat().st_size
            if size == 0:
                raise ValueError(f"Downloaded file is empty: {dest}")
            print(f"  Saved to {dest} ({size:,} bytes)")
            return
        except (urllib.error.URLError, ValueError) as e:
            if attempt < retries:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                print(f"  Error: Failed to download {url}: {e}", file=sys.stderr)
                sys.exit(1)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching raw emoji data sources...")
    for filename, url in SOURCES.items():
        dest = RAW_DIR / filename
        download(url, dest)
    print("Done.")


if __name__ == "__main__":
    main()
