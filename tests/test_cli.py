"""Tests for the Click CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from emojify.cli import cli, _is_emoji_input


def _create_test_index_files() -> tuple[Path, Path]:
    """Create a small fake index and metadata in a temp directory.

    Returns (metadata_path, index_path).
    """
    tmpdir = tempfile.mkdtemp()
    metadata_path = Path(tmpdir) / "emoji_metadata.json"
    index_path = Path(tmpdir) / "emoji_index.npz"

    n = 10
    dim = 1536
    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, dim).astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    categories = [
        "Smileys & Emotion",
        "People & Body",
        "Animals & Nature",
        "Food & Drink",
        "Travel & Places",
        "Activities",
        "Objects",
        "Symbols",
        "Flags",
        "Smileys & Emotion",
    ]
    emoji_chars = [chr(0x1F600 + i) for i in range(n)]
    metadata = [
        {
            "emoji": emoji_chars[i],
            "short_name": f"emoji_{i}",
            "keywords": [f"kw_{i}_a", f"kw_{i}_b"],
            "category": categories[i % len(categories)],
            "description": f"Emoji {i} description",
        }
        for i in range(n)
    ]

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    np.savez(
        index_path,
        embeddings=embeddings,
        emoji_list=np.array(emoji_chars, dtype=object),
    )

    return metadata_path, index_path


class TestIsEmojiInput:
    """Tests for the _is_emoji_input heuristic."""

    def test_pure_emoji(self):
        assert _is_emoji_input("🍕🍺📺") is True

    def test_pure_text(self):
        assert _is_emoji_input("hello world") is False

    def test_mixed_mostly_emoji(self):
        assert _is_emoji_input("🍕🍺🎉 yes") is True

    def test_mixed_mostly_text(self):
        assert _is_emoji_input("hello world 🍕") is False

    def test_empty(self):
        assert _is_emoji_input("") is False

    def test_whitespace_only(self):
        assert _is_emoji_input("   ") is False


class TestVersionCommand:
    """Tests for the version command."""

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "emojify v0.1.0" in result.output


class TestTextCommand:
    """Tests for the text command."""

    @patch("emojify.cli._validate_startup")
    @patch("emojify.text_to_emoji.get_embedding")
    def test_text_command(self, mock_embed, mock_validate):
        """text command returns results with emoji and scores."""
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        metadata_path, index_path = _create_test_index_files()

        runner = CliRunner()
        with patch("emojify.cli._get_index") as mock_get_index:
            from emojify.index import EmojiIndex

            mock_get_index.return_value = EmojiIndex(
                metadata_path=metadata_path, index_path=index_path,
            )
            result = runner.invoke(cli, ["text", "happy times"])

        assert result.exit_code == 0
        # Output should contain score values (decimal numbers)
        assert "0." in result.output
        # Output should contain emoji names
        assert "emoji_" in result.output

    @patch("emojify.cli._validate_startup")
    @patch("emojify.text_to_emoji.get_embedding")
    def test_text_top_k(self, mock_embed, mock_validate):
        """text command respects --top-k flag."""
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        metadata_path, index_path = _create_test_index_files()

        runner = CliRunner()
        with patch("emojify.cli._get_index") as mock_get_index:
            from emojify.index import EmojiIndex

            mock_get_index.return_value = EmojiIndex(
                metadata_path=metadata_path, index_path=index_path,
            )
            result = runner.invoke(cli, ["text", "--top-k", "3", "test query"])

        assert result.exit_code == 0

    @patch("emojify.cli._validate_startup")
    @patch("emojify.text_to_emoji.get_embedding")
    def test_text_no_diversity(self, mock_embed, mock_validate):
        """text command with --no-diversity skips suggested sequence."""
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        metadata_path, index_path = _create_test_index_files()

        runner = CliRunner()
        with patch("emojify.cli._get_index") as mock_get_index:
            from emojify.index import EmojiIndex

            mock_get_index.return_value = EmojiIndex(
                metadata_path=metadata_path, index_path=index_path,
            )
            result = runner.invoke(cli, ["text", "--no-diversity", "test"])

        assert result.exit_code == 0
        assert "Suggested sequence" not in result.output


class TestSuggestCommand:
    """Tests for the suggest command."""

    @patch("emojify.cli._validate_startup")
    @patch("emojify.text_to_emoji.get_embedding")
    def test_suggest_command(self, mock_embed, mock_validate):
        """suggest command returns a string of emoji characters."""
        mock_embed.return_value = np.random.RandomState(99).randn(1536)

        metadata_path, index_path = _create_test_index_files()

        runner = CliRunner()
        with patch("emojify.cli._get_index") as mock_get_index:
            from emojify.index import EmojiIndex

            mock_get_index.return_value = EmojiIndex(
                metadata_path=metadata_path, index_path=index_path,
            )
            result = runner.invoke(cli, ["suggest", "good morning"])

        assert result.exit_code == 0
        output = result.output.strip()
        # Output should be emoji characters (high Unicode)
        assert len(output) >= 1
        for char in output:
            assert ord(char) > 127


class TestDecodeCommand:
    """Tests for the decode command."""

    @patch("emojify.cli._validate_startup")
    def test_decode_no_llm(self, mock_validate):
        """decode --no-llm shows individual meanings without LLM call."""
        metadata_path, index_path = _create_test_index_files()

        runner = CliRunner()
        with patch("emojify.cli._get_index") as mock_get_index:
            from emojify.index import EmojiIndex

            idx = EmojiIndex(
                metadata_path=metadata_path, index_path=index_path,
            )
            mock_get_index.return_value = idx

            # Use an emoji from our test index
            test_emoji = idx.metadata[0]["emoji"]
            result = runner.invoke(cli, ["decode", "--no-llm", test_emoji])

        assert result.exit_code == 0
        assert "Individual meanings" in result.output

    @patch("emojify.cli._validate_startup")
    @patch("emojify.decoder.openai.ChatCompletion.create")
    @patch("emojify.decoder.get_openai_api_key", return_value="fake-key")
    def test_decode_with_llm(self, mock_key, mock_chat, mock_validate):
        """decode command shows both individual meanings and combined interpretation."""
        mock_chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test interpretation"))]
        )

        metadata_path, index_path = _create_test_index_files()

        runner = CliRunner()
        with patch("emojify.cli._get_index") as mock_get_index:
            from emojify.index import EmojiIndex

            idx = EmojiIndex(
                metadata_path=metadata_path, index_path=index_path,
            )
            mock_get_index.return_value = idx

            test_emoji = idx.metadata[0]["emoji"]
            result = runner.invoke(cli, ["decode", test_emoji])

        assert result.exit_code == 0
        assert "Individual meanings" in result.output
        assert "Combined interpretation" in result.output

    @patch("emojify.cli._validate_startup")
    def test_decode_no_emoji_in_input(self, mock_validate):
        """decode with plain text shows a 'no emoji found' message."""
        metadata_path, index_path = _create_test_index_files()

        runner = CliRunner()
        with patch("emojify.cli._get_index") as mock_get_index:
            from emojify.index import EmojiIndex

            mock_get_index.return_value = EmojiIndex(
                metadata_path=metadata_path, index_path=index_path,
            )
            result = runner.invoke(cli, ["decode", "hello world"])

        assert result.exit_code == 0
        assert "No emoji found" in result.output


class TestMissingApiKey:
    """Tests for missing API key error handling."""

    def test_text_missing_api_key(self):
        """text command exits with error when OPENAI_API_KEY is unset."""
        runner = CliRunner()
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(cli, ["text", "hello"])

        assert result.exit_code != 0

    def test_suggest_missing_api_key(self):
        """suggest command exits with error when OPENAI_API_KEY is unset."""
        runner = CliRunner()
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(cli, ["suggest", "hello"])

        assert result.exit_code != 0
