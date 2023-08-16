"""Emoji-to-text decoder — look up meanings and generate interpretations.

Two-stage pipeline:
  Stage 1 (deterministic): Parse the emoji string into individual characters,
           look up each one's metadata (short name, keywords, description).
  Stage 2 (LLM-assisted): Concatenate descriptions and prompt GPT-3.5 Turbo
           to generate a natural-language interpretation of the sequence.
"""

import re
from dataclasses import dataclass, field

import openai

from emojify.config import DECODE_MODEL, get_openai_api_key
from emojify.index import EmojiIndex


# Regex pattern matching emoji characters including:
#   - Basic emoji (U+1F600–U+1F64F, U+1F300–U+1F5FF, etc.)
#   - Flags (regional indicators U+1F1E0–U+1F1FF)
#   - Skin-tone modifiers (U+1F3FB–U+1F3FF)
#   - ZWJ sequences joined by U+200D
#   - Variation selectors (U+FE0F)
#   - Miscellaneous symbols (U+2600–U+27BF, U+2300–U+23FF)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001F1E0-\U0001F1FF"  # regional indicators (flags)
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002300-\U000023FF"  # misc technical
    "\U0000200D"             # ZWJ
    "\U0000FE0F"             # variation selector
    "\U0001F3FB-\U0001F3FF"  # skin tone modifiers
    "]+",
    flags=re.UNICODE,
)


@dataclass
class EmojiDescription:
    """Metadata for a single emoji character from Stage 1 lookup."""

    emoji: str
    short_name: str
    keywords: list[str]
    description: str
    found_in_index: bool = True


@dataclass
class DecodeResult:
    """Complete decode output for an emoji string."""

    individual: list[EmojiDescription] = field(default_factory=list)
    combined_interpretation: str = ""


def _parse_emoji(text: str) -> list[str]:
    """Split an emoji string into individual emoji (handling ZWJ sequences).

    Returns a list of emoji strings.  Multi-codepoint emoji like flags
    (🇺🇸) and ZWJ sequences (👨‍💻) are kept as single entries.
    """
    # Find all contiguous emoji runs
    raw_matches = _EMOJI_RE.findall(text)

    result: list[str] = []
    for chunk in raw_matches:
        # Strip variation selectors for cleaner matching
        clean = chunk.replace("\uFE0F", "")
        if not clean:
            continue

        # Split on ZWJ boundaries to detect ZWJ sequences
        # A ZWJ sequence is characters joined by U+200D — keep those together
        if "\u200D" in clean:
            # This is a ZWJ sequence — keep it as one unit
            result.append(chunk)
        else:
            # Check for flag sequences (pairs of regional indicators)
            i = 0
            while i < len(clean):
                cp = ord(clean[i])
                # Regional indicator symbols form flag pairs
                if 0x1F1E0 <= cp <= 0x1F1FF and i + 1 < len(clean):
                    next_cp = ord(clean[i + 1])
                    if 0x1F1E0 <= next_cp <= 0x1F1FF:
                        result.append(clean[i : i + 2])
                        i += 2
                        continue
                # Check for skin-tone modifier following an emoji
                if (
                    i + 1 < len(clean)
                    and 0x1F3FB <= ord(clean[i + 1]) <= 0x1F3FF
                ):
                    result.append(clean[i : i + 2])
                    i += 2
                    continue
                # Skip standalone skin-tone modifiers
                if 0x1F3FB <= cp <= 0x1F3FF:
                    i += 1
                    continue
                result.append(clean[i])
                i += 1

    return result


def _build_prompt(descriptions: list[EmojiDescription]) -> str:
    """Build the GPT-3.5 Turbo prompt from Stage 1 results."""
    lines = []
    for desc in descriptions:
        lines.append(f"{desc.emoji}: {desc.description}")

    emoji_section = "\n".join(lines)

    return (
        f"Given these emoji and their meanings:\n"
        f"{emoji_section}\n\n"
        f"What is the most likely message someone intended when sending "
        f"this emoji sequence?\n"
        f"Respond with a single, natural sentence."
    )


def decode_emoji(
    emoji_string: str,
    index: EmojiIndex,
    use_llm: bool = True,
) -> DecodeResult:
    """Decode an emoji string into natural-language meaning.

    Stage 1 (deterministic): Parse individual emoji, look up metadata.
    Stage 2 (optional LLM): Generate a combined interpretation via GPT-3.5.

    Args:
        emoji_string: A string of one or more emoji characters.
        index: Preloaded EmojiIndex with metadata.
        use_llm: If True, call GPT-3.5 Turbo for combined interpretation.
                 If False, only return Stage 1 (individual lookups).

    Returns:
        DecodeResult with individual descriptions and optionally a
        combined interpretation.
    """
    if not emoji_string or not emoji_string.strip():
        return DecodeResult()

    # --- Stage 1: Parse and look up ---
    parsed = _parse_emoji(emoji_string)
    if not parsed:
        return DecodeResult()

    individual: list[EmojiDescription] = []
    for emoji_char in parsed:
        entry = index.lookup(emoji_char)
        if entry is not None:
            individual.append(
                EmojiDescription(
                    emoji=emoji_char,
                    short_name=entry["short_name"],
                    keywords=entry["keywords"],
                    description=entry["description"],
                    found_in_index=True,
                )
            )
        else:
            individual.append(
                EmojiDescription(
                    emoji=emoji_char,
                    short_name="unknown emoji",
                    keywords=[],
                    description=f"{emoji_char} (unknown emoji)",
                    found_in_index=False,
                )
            )

    result = DecodeResult(individual=individual)

    # --- Stage 2: LLM interpretation ---
    if use_llm and individual:
        prompt = _build_prompt(individual)

        if not openai.api_key:
            openai.api_key = get_openai_api_key()

        response = openai.ChatCompletion.create(
            model=DECODE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
        )
        result.combined_interpretation = response.choices[0].message.content.strip()

    return result
