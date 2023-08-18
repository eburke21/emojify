# emojify Evaluation Suite

## Methodology

### Dataset

50 manually curated test cases split across two pipelines:

- **30 text_to_emoji cases** across 10 semantic categories (greeting, emotion, activity, reaction, celebration, frustration, food, work, weather, achievement) — 3 cases per category
- **20 emoji_to_text cases** across 10 semantic categories (celebration, emotion, travel, contemplation, approval, food, work, weather, sarcasm, abstract) — 2 cases per category

Each text_to_emoji case specifies 3 acceptable emoji. Each emoji_to_text case specifies 2-3 acceptable natural language interpretations.

The dataset includes deliberate hard cases:
- **Sarcasm:** "oh great, another meeting" (text), 🙃🔫 (emoji)
- **Slang:** "that's fire"
- **Ambiguous sequences:** 🏃✈️⏰ (running late? traveling to a race?)
- **Abstract concepts:** ⏳💭🌅, 🌍❤️🤝

### Scoring Rubric

**Text to Emoji (`score_text_to_emoji`):**
| Score | Criteria |
|-------|----------|
| 3 (Perfect) | Any expected emoji appears in the top-3 results |
| 2 (Reasonable) | Any expected emoji appears in top-5 (but not top-3) |
| 1 (Miss) | No expected emoji in top-5 |

**Emoji to Text (`score_emoji_to_text`):**
| Score | Criteria |
|-------|----------|
| 3 (Perfect) | Cosine similarity between interpretation and best acceptable string > 0.85 |
| 2 (Reasonable) | Cosine similarity > 0.70 |
| 1 (Miss) | Cosine similarity <= 0.70 |

The emoji-to-text scoring uses ada-002 embeddings of both the LLM-generated interpretation and each acceptable answer, taking the maximum cosine similarity across all acceptable strings.

### Running the Evaluation

```bash
export OPENAI_API_KEY='your-key-here'
make eval
```

Results are saved to `results/eval_run.json`.

**Target:** Average score >= 2.3 across all 50 cases.

---

## Results

> Update this section after running `make eval`.

### Overall

- Average score: _TBD_ / 3.00
- Target (>= 2.3): _TBD_
- Distribution: 3=_TBD_ | 2=_TBD_ | 1=_TBD_

### Per-Category Breakdown

_Insert table from eval output._

---

## Failure Mode Analysis

> Update this section after reviewing worst cases from eval output.

### Expected Patterns (from spec)

1. **Concrete nouns score highest** — food, animals, weather should average 2.7+. ada-002 embeddings are excellent at mapping "pizza" near the pizza emoji.
2. **Abstract emotions are harder** — sarcasm, nuance should average ~2.0. "That meeting was brutal" maps well to 😤 but misses the sarcasm layer.
3. **Compound sequences are the hardest** — multi-emoji narrative ordering is ambiguous. 🏃✈️⏰ could be "running late for a flight" or "traveling to a race."
4. **The diversity filter matters** — without it, "I'm so happy" returns 😀😃😄😁😊 (five nearly identical smiling faces). With diversity filtering, it returns 😊🎉✨.

### Observed Patterns

_Update after running eval._

---

## What Would Improve Results

1. **Enriched descriptions** — Add usage context to emoji descriptions ("often used sarcastically", "commonly paired with 💀 for humor"). The description template is simple; richer descriptions would give the embedding model more semantic signal.
2. **Fine-tuned embeddings** — A model trained specifically on emoji-text pairs rather than general text would capture emoji-specific semantics better.
3. **Few-shot prompting for decode** — Include 2-3 examples in the GPT-3.5 Turbo prompt for better emoji-to-text interpretations, especially for ambiguous or sarcastic sequences.
4. **Larger eval set** — 50 cases is a starting point; 200+ cases would give more statistical power for per-category analysis and would reduce the impact of individual outliers.

---

## Lessons Learned

1. **"Embedding quality depends more on what you embed than which model you use."** Generated descriptions significantly outperform raw keywords for embedding quality.
2. **"Not everything needs a vector database."** At 1,800 vectors, brute-force numpy cosine similarity takes <5ms.
3. **"Cross-modal search is just regular search in the right embedding space."** The same principle behind CLIP, cross-lingual retrieval, and multimodal search.
4. **"Evaluation is never optional, even for a fun project."** This eval set reveals category-specific weaknesses that manual spot-checking would miss.
