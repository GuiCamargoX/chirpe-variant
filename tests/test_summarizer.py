"""Tests for the SmolLM2 lightweight summarizer.

The fast tests cover the pure prompt-building and output-cleaning helpers and
do not download or instantiate any model. The integration test that loads the
real SmolLM2 checkpoint is opt-in (it downloads weights from the Hub) and only
runs when ``CHIRPE_RUN_SMOLLM2=1`` is set.
"""

import os

import pytest

from chirpe.data.summarizer import SMOLLM2_INSTRUCTION, SmolLM2Summarizer


class TestSmolLM2Helpers:
    """Fast tests for the model-free helpers (no weights downloaded)."""

    def test_build_messages_structure(self):
        """Messages carry the instruction as system and the segment as user."""
        messages = SmolLM2Summarizer._build_messages("Patient hears voices.")

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": SMOLLM2_INSTRUCTION}
        assert messages[1]["role"] == "user"
        assert "Patient hears voices." in messages[1]["content"]

    def test_clean_output_strips_whitespace(self):
        """Surrounding whitespace is removed."""
        assert SmolLM2Summarizer._clean_output("  The interviewee is well.  ") == (
            "The interviewee is well."
        )

    @pytest.mark.parametrize(
        "marker",
        ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    )
    def test_clean_output_trims_chat_markers(self, marker):
        """Generation is cut at the first leaked chat-template marker."""
        raw = f"A short summary.{marker}\nleaked continuation"
        assert SmolLM2Summarizer._clean_output(raw) == "A short summary."

    def test_clean_output_keeps_clean_text(self):
        """Text without markers is returned unchanged (apart from stripping)."""
        text = "The interviewee reports mild concentration difficulties."
        assert SmolLM2Summarizer._clean_output(text) == text

    def test_default_model_is_lightweight_instruct(self):
        """The default checkpoint is the small SmolLM2 instruct variant."""
        assert SmolLM2Summarizer.DEFAULT_MODEL_NAME == "HuggingFaceTB/SmolLM2-360M-Instruct"


@pytest.mark.skipif(
    os.environ.get("CHIRPE_RUN_SMOLLM2") != "1",
    reason="Set CHIRPE_RUN_SMOLLM2=1 to run the SmolLM2 download/inference test.",
)
class TestSmolLM2Integration:
    """Opt-in test that loads the real (small) SmolLM2 model and summarizes."""

    def test_summarize_segment_returns_text(self):
        summarizer = SmolLM2Summarizer(device="cpu", max_new_tokens=48)
        segment = (
            "Interviewer: Have you been hearing things others cannot hear?\n"
            "Interviewee: Sometimes at night I hear whispering when no one is there, "
            "and it has made it hard to sleep or focus at work."
        )
        summary = summarizer.summarize_segment(segment)

        assert isinstance(summary, str)
        assert summary.strip()
        # No chat-template markers should survive cleaning.
        assert "<|im_end|>" not in summary
        assert "<|im_start|>" not in summary
