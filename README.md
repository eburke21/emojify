# 🔁 emojify

A CLI tool that translates between natural language and emoji sequences in both directions. Type a sentence and get the perfect emoji sequence, or paste a string of emoji and get the most likely intended message back.

Built with OpenAI's `text-embedding-ada-002` model and a curated dataset of 2,300+ emoji descriptions, mapped into a shared vector space where cosine similarity does the matching.

## Quick Demo

```bash
$ emojify text "just deployed to production at 2am"
┌────────┬──────────┬─────────────────┬──────────────────────────────────┐
│ Emoji  │    Score │ Name            │ Keywords                         │
├────────┼──────────┼─────────────────┼──────────────────────────────────┤
│ 🚀     │     0.89 │ rocket          │ launch, ship, deploy             │
│ 😴     │     0.85 │ sleeping face   │ tired, sleepy, exhausted         │
│ 💻     │     0.84 │ laptop          │ computer, code, programming      │
│ 🌙     │     0.83 │ crescent moon   │ night, late, evening             │
│ ⚡     │     0.82 │ high voltage    │ fast, quick, electric            │
└────────┴──────────┴─────────────────┴──────────────────────────────────┘

Suggested sequence: 🚀💻🌙

$ emojify suggest "good morning, hope you have a great day"
☀️🌅😊💛✨

$ emojify decode "🎉🍕🎂"
Individual meanings:
  🎉  party popper — celebration, congratulations, hooray
  🍕  pizza — food, slice, italian
  🎂  birthday cake — birthday, celebration, candles

Combined interpretation: "Happy birthday! Let's celebrate with pizza."
```

## How It Works

Every emoji has a curated natural language description (e.g., "party popper -- commonly used to express: celebration, congratulations, hooray"). Both these descriptions and your query text are embedded into the same 1,536-dimensional vector space using OpenAI's `text-embedding-ada-002`. The emoji whose descriptions are closest to your query -- measured by cosine similarity -- are the best matches. No fine-tuning, no classification head, no prompt engineering. Just the raw geometry of embedding space.

The emoji-to-text direction works in reverse: each emoji in your input is looked up in a metadata index to retrieve its description and keywords, then GPT-3.5 Turbo generates a natural language interpretation from the combined descriptions. A diversity filter prevents redundant sequences (e.g., five nearly identical smiling faces) by enforcing category variety in the results.

## Installation

```bash
# Clone and install
git clone https://github.com/your-username/emojify.git
cd emojify
pip install -e ".[dev]"

# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Build the data pipeline and embedding index
make fetch-data        # Download Unicode CLDR + Emojilib data
make build-metadata    # Merge sources, generate descriptions
make build-index       # Embed all descriptions via ada-002 (~$0.02)
```

## Usage

### `emojify text` -- Find the best emoji for a query

```bash
$ emojify text "I love pizza"
# Rich table with top-5 emoji ranked by similarity score
# Plus a diversity-filtered "Suggested sequence" of 3 emoji

$ emojify text "happy birthday" --top-k 10     # More results
$ emojify text "hello" --no-diversity           # Skip suggested sequence
$ emojify text "hello" --verbose                # Show timing info
```

### `emojify suggest` -- Quick emoji sequence for copy-paste

```bash
$ emojify suggest "the meeting was absolutely brutal"
😤💀📊⏰😵

$ emojify suggest "pizza night" --count 3
🍕🌙😋
```

### `emojify decode` -- Translate emoji to text

```bash
$ emojify decode "🍕🍺📺🏈"
Individual meanings:
  🍕  pizza — cheese, food, slice
  🍺  beer mug — beer, drink, bar
  📺  television — tv, watch, screen
  🏈  american football — football, sports, nfl

Combined interpretation: "Watching football with pizza and beer."

$ emojify decode "🚀🔥💯" --no-llm    # Stage 1 only, no API call
```

### `emojify interactive` -- REPL mode

```bash
$ emojify interactive
emojify v0.1.0 — type text or paste emoji (q to quit)

> I love sushi
🍣❤️🇯🇵

> 🌧️☕📖
"Rainy day, reading a book with coffee."

> running late to the airport
🏃✈️⏰😰

> q
Goodbye! 👋
```

Auto-detects whether your input is text or emoji and routes to the right pipeline.

## Evaluation

A 50-case eval suite scores both pipelines on a 1-3 scale:

- **Text to emoji:** 3 if any expected emoji in top-3, 2 if in top-5, 1 otherwise
- **Emoji to text:** 3 if cosine similarity to acceptable interpretation > 0.85, 2 if > 0.70, 1 otherwise

Target average score: >= 2.3. Run with `make eval`. See [tests/eval/README.md](tests/eval/README.md) for methodology and findings.

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11+ | Standard for ML/AI tooling |
| CLI | click | Clean subcommand structure |
| Terminal Output | rich | Colored emoji, tables, progress bars |
| Embeddings | text-embedding-ada-002 | Best cost/quality ratio ($0.0001/1K tokens) |
| LLM (decode) | GPT-3.5 Turbo | Fast, cheap single-sentence generation |
| Vector Math | numpy | Cosine similarity -- no vector DB needed at this scale |
| Caching | sqlite3 (stdlib) | Cache query embeddings to avoid redundant API calls |
| Data Sources | Unicode CLDR + Emojilib | Canonical names + community keywords for 2,300+ emoji |
| Testing | pytest | 76 unit tests, all API calls mocked |

## Design Decisions

**Why no vector database?** At ~2,300 vectors of dimension 1,536, the entire index fits in ~14MB of RAM. A brute-force numpy cosine similarity scan takes <5ms. ChromaDB, Pinecone, or FAISS would add complexity with zero benefit at this scale. Knowing when *not* to use a tool is the point.

**Why no LangChain?** The pipeline is: embed, compare, rank. Three numpy operations. Wrapping this in LangChain abstractions would obscure the mechanics that this project is designed to demonstrate.

**Why GPT-3.5 Turbo instead of GPT-4 for decode?** The decode task is lightweight -- the context is a few emoji descriptions, and the output is a single sentence. GPT-3.5 Turbo is 20x cheaper and fast enough for a CLI tool.

**Why ada-002 instead of newer models?** This project targets September 2023, when ada-002 was the standard embedding model. The architecture is model-agnostic -- swapping in a newer model requires changing one config constant.

## Project Structure

```
emojify/
├── src/emojify/
│   ├── cli.py               # Click CLI (text, suggest, decode, interactive)
│   ├── embeddings.py         # OpenAI embedding API + SQLite cache
│   ├── index.py              # EmojiIndex class (load, cosine similarity search)
│   ├── text_to_emoji.py      # Text → emoji pipeline with diversity filtering
│   ├── decoder.py            # Emoji → text pipeline (metadata lookup + LLM)
│   ├── diversity.py          # Category-based diversity filter
│   ├── eval.py               # Scoring functions for evaluation suite
│   └── config.py             # API key, model constants, data paths
├── data/
│   ├── raw/                  # Downloaded Unicode CLDR + Emojilib data
│   ├── processed/            # emoji_metadata.json + emoji_index.npz
│   └── scripts/              # Data pipeline (fetch, merge, describe, embed)
├── tests/
│   ├── eval/                 # 50-case evaluation suite + runner
│   ├── test_cli.py           # CLI tests (16 tests)
│   ├── test_decoder.py       # Decoder tests (14 tests)
│   ├── test_diversity.py     # Diversity filter tests (6 tests)
│   ├── test_embeddings.py    # Embedding + cache tests (7 tests)
│   ├── test_eval.py          # Eval scoring tests (14 tests)
│   ├── test_index.py         # Index search tests (8 tests)
│   └── test_text_to_emoji.py # Text-to-emoji pipeline tests (5 tests)
├── pyproject.toml
├── Makefile
└── README.md
```
