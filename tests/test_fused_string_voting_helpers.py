"""Tests for fixed-slot fused string voting helpers."""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxruntime_extensions")

from scripts.onnx.build_fused_string_voting_onnx import (
    build_slot_inputs,
    numpy_voting_reference,
)


def test_build_slot_inputs_pads_text_and_mask():
    """Short transcript segment lists should be padded and masked out."""
    text, mask = build_slot_inputs(["first", "second"], max_segments=5)

    assert text.tolist() == ["first", "second", "", "", ""]
    assert mask.dtype == np.float32
    assert mask.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_build_slot_inputs_truncates_overflow_segments():
    """Only the first max_segments strings should be retained."""
    text, mask = build_slot_inputs(["a", "b", "c", "d"], max_segments=3)

    assert text.tolist() == ["a", "b", "c"]
    assert mask.tolist() == [1.0, 1.0, 1.0]


def test_numpy_voting_reference_ignores_masked_slots():
    """Transcript outputs should be computed only from active segment slots."""
    logits = np.array(
        [
            [0.0, 3.0],
            [3.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )
    mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    reference = numpy_voting_reference(logits, mask)
    active_probs = reference["probabilities"][:2]
    expected_transcript_probs = active_probs.mean(axis=0)

    assert np.allclose(reference["transcript_probabilities"], expected_transcript_probs)
    assert reference["transcript_label_majority"] == 0
