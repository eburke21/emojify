"""Tests for the emoji-to-text decoder."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from emojify.index import EmojiIndex
from emojify.decoder import (
    decode_emoji,
    _parse_emoji,
    _build_prompt,
    EmojiDescription,
    DecodeResult,
)


def _create_test_index() -> EmojiIndex:
    """Create a small EmojiIndex with known emoji for decoder testing."""
    tmpdir = tempfile.mkdtemp()
    metadata_path = Path(tmpdir) / "metadata.json"
    index_path = Path(tmpdir) / "index.npz"

    emoji_chars = ["🍕", "🍺", "📺", "🎉", "🎂"]
    n = len(emoji_chars)
    dim = 1536

    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, dim).astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    metadata = [
        {
            "emoji": "🍕",
            "short_name": "pizza",
            "keywords": ["cheese", "food", "slice"],
            "category": "Food & Drink",
            "description": "🍕 (pizza) — commonly used to express: cheese, food, slice",
        },
        {
            "emoji": "🍺",
            "short_name": "beer mug",
            "keywords": ["beer", "drink", "bar"],
            "category": "Food & Drink",
            "description": "🍺 (beer mug) — commonly used to express: beer, drink, bar",
        },
        {
            "emoji": "📺",
            "short_name": "television",
            "keywords": ["tv", "show", "watch"],
            "category": "Objects",
            "description": "📺 (television) — commonly used to express: tv, show, watch",
        },
        {
            "emoji": "🎉",
            "short_name": "party popper",
            "keywords": ["celebration", "party", "hooray"],
            "category": "Activities",
            "description": "🎉 (party popper) — commonly used to express: celebration, party, hooray",
        },
        {
            "emoji": "🎂",
            "short_name": "birthday cake",
            "keywords": ["birthday", "cake", "candles"],
            "category": "Food & Drink",
            "description": "🎂 (birthday cake) — commonly used to express: birthday, cake, candles",
        },
    ]

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    np.savez(
        index_path,
        embeddings=embeddings,
        emoji_list=np.array(emoji_chars, dtype=object),
    )

    return EmojiIndex(metadata_path=metadata_path, index_path=index_path)


class TestParseEmoji:
    """Tests for the _parse_emoji helper."""

    def test_single_emoji(self):
        result = _parse_emoji("🍕")
        assert result == ["🍕"]

    def test_multiple_emoji(self):
        result = _parse_emoji("🍕🍺📺")
        assert result == ["🍕", "🍺", "📺"]

    def test_empty_string(self):
        result = _parse_emoji("")
        assert result == []

    def test_plain_text_ignored(self):
        result = _parse_emoji("hello world")
        assert result == []

    def test_mixed_text_and_emoji(self):
        result = _parse_emoji("I love 🍕 and 🍺")
        assert result == ["🍕", "🍺"]


class TestStage1Lookup:
    """Tests for Stage 1 — deterministic metadata lookup."""

    def setup_method(self):
        self.index = _create_test_index()

    def test_known_emoji_lookup(self):
        """Stage 1 populates descriptions for known emoji."""
        result = decode_emoji("🍕🍺📺", self.index, use_llm=False)

        assert len(result.individual) == 3
        assert result.individual[0].emoji == "🍕"
        assert result.individual[0].short_name == "pizza"
        assert result.individual[0].found_in_index is True
        assert "cheese" in result.individual[0].keywords

        assert result.individual[1].emoji == "🍺"
        assert result.individual[1].short_name == "beer mug"

        assert result.individual[2].emoji == "📺"
        assert result.individual[2].short_name == "television"

    def test_no_llm_interpretation(self):
        """With use_llm=False, combined_interpretation is empty."""
        result = decode_emoji("🍕", self.index, use_llm=False)
        assert result.combined_interpretation == ""

    def test_unknown_emoji_handled(self):
        """Unknown emoji gets a fallback description, doesn't crash."""
        result = decode_emoji("🤖", self.index, use_llm=False)

        assert len(result.individual) == 1
        assert result.individual[0].emoji == "🤖"
        assert result.individual[0].short_name == "unknown emoji"
        assert result.individual[0].found_in_index is False

    def test_empty_input(self):
        """Empty string returns empty DecodeResult."""
        result = decode_emoji("", self.index, use_llm=False)
        assert result.individual == []
        assert result.combined_interpretation == ""

    def test_mixed_known_unknown(self):
        """Mix of known and unknown emoji both handled correctly."""
        result = decode_emoji("🍕🤖🎉", self.index, use_llm=False)

        assert len(result.individual) == 3
        assert result.individual[0].found_in_index is True
        assert result.individual[1].found_in_index is False
        assert result.individual[2].found_in_index is True


class TestStage2LLM:
    """Tests for Stage 2 — LLM-based interpretation."""

    def setup_method(self):
        self.index = _create_test_index()

    @patch("openai.OpenAI")
    @patch("emojify.decoder.get_openai_api_key", return_value="fake-key")
    def test_llm_call_made(self, mock_key, mock_openai_cls):
        """Stage 2 calls GPT-3.5 Turbo and returns the interpretation."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Pizza and beer night!"))]
        )

        result = decode_emoji("🍕🍺", self.index, use_llm=True)

        assert result.combined_interpretation == "Pizza and beer night!"
        mock_client.chat.completions.create.assert_called_once()

        # Verify the prompt contains the emoji descriptions
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        prompt = messages[0]["content"]
        assert "🍕" in prompt
        assert "🍺" in prompt
        assert "pizza" in prompt

    @patch("openai.OpenAI")
    @patch("emojify.decoder.get_openai_api_key", return_value="fake-key")
    def test_prompt_format(self, mock_key, mock_openai_cls):
        """The prompt is correctly constructed from emoji descriptions."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="test response"))]
        )

        decode_emoji("🎉🎂", self.index, use_llm=True)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        prompt = messages[0]["content"]

        assert "Given these emoji and their meanings:" in prompt
        assert "party popper" in prompt
        assert "birthday cake" in prompt
        assert "single, natural sentence" in prompt

    @patch("openai.OpenAI")
    @patch("emojify.decoder.get_openai_api_key", return_value="fake-key")
    def test_model_parameters(self, mock_key, mock_openai_cls):
        """Verify GPT-3.5 Turbo is called with correct parameters."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))]
        )

        decode_emoji("🍕", self.index, use_llm=True)

        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-3.5-turbo"
        assert call_args["max_tokens"] == 100
        assert call_args["temperature"] == 0.3


class TestBuildPrompt:
    """Tests for the prompt builder."""

    def test_prompt_structure(self):
        descriptions = [
            EmojiDescription(
                emoji="🍕", short_name="pizza",
                keywords=["food"], description="🍕 (pizza) — food",
            ),
        ]
        prompt = _build_prompt(descriptions)

        assert "Given these emoji" in prompt
        assert "🍕: 🍕 (pizza) — food" in prompt
        assert "single, natural sentence" in prompt
