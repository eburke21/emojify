# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Emojify is a bidirectional CLI tool that translates between natural language and emoji sequences using OpenAI's `text-embedding-ada-002` embeddings. Text→emoji uses cosine similarity search over ~1,800 precomputed emoji embeddings; emoji→text uses metadata lookup + GPT-3.5 Turbo interpretation.

## Commands

```bash
make install           # pip install -e ".[dev]"
make test              # python -m pytest tests/ -v
make eval              # Run 50-case evaluation suite (requires API key)
make fetch-data        # Download Unicode CLDR + Emojilib source data
make build-metadata    # Parse & merge sources, generate descriptions
make build-index       # Embed all descriptions via ada-002 (~$0.02)
```

Run a single test:
```bash
python -m pytest tests/test_cli.py -v
python -m pytest tests/test_cli.py::test_name -v
```

## Architecture

**Two pipelines:**

1. **Text → Emoji** (`text_to_emoji.py`): Embed query via ada-002 → cosine similarity search against `emoji_index.npz` → diversity filter (max 1 per Unicode category) → ranked `EmojiMatch` list

2. **Emoji → Text** (`decoder.py`): Stage 1 (deterministic) parses emoji and looks up metadata in index → Stage 2 (LLM) sends descriptions to GPT-3.5 Turbo for natural language interpretation

**Key modules in `src/emojify/`:**
- `index.py` — `EmojiIndex` class: loads embeddings + metadata, provides `search()` and `lookup()`
- `embeddings.py` — OpenAI API calls + SQLite cache for query embeddings
- `diversity.py` — Category-based deduplication to prevent redundant emoji sequences
- `cli.py` — Click CLI with subcommands: `text`, `suggest`, `decode`, `interactive`, `version`
- `config.py` — Model constants, data paths, API key loading from `OPENAI_API_KEY` env var
- `eval.py` — Scoring functions for evaluation suite

**Data pipeline** (`data/scripts/`): `fetch_data.py` → `merge_sources.py` → `generate_descriptions.py` → `build_index.py`. Sources are Unicode CLDR (`en.xml`) + Emojilib (`emojilib.json`), merged into `emoji_metadata.json`, then embedded into `emoji_index.npz`.

## Key Design Decisions

- **No vector database**: ~1,800 × 1,536-dim embeddings; brute-force numpy cosine similarity takes <5ms
- **OpenAI SDK pinned to <1.0** (`openai>=0.28,<1.0`): uses the older API style
- **Python 3.11+** required (uses `|` union type syntax)
- **All tests mock API calls**: no real OpenAI calls in tests; 70+ unit tests total
- **SQLite cache** for query embeddings avoids redundant API calls

## Evaluation

50 manually curated test cases in `tests/eval/test_cases.yaml` (30 text→emoji, 20 emoji→text). Scoring: 3=top-3 match, 2=top-5, 1=miss for text→emoji; cosine similarity thresholds for emoji→text. Target: average ≥ 2.3/3.0.
