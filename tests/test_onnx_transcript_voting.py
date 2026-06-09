"""Tests for transcript-level voting ONNX graph augmentation."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper

from scripts.onnx.add_transcript_voting_onnx import (
    SEGMENT_LABELS,
    SEGMENT_PROBABILITIES,
    TRANSCRIPT_LABEL_AVERAGE,
    TRANSCRIPT_PROBABILITIES,
    add_transcript_voting_outputs,
)


def make_logits_identity_model():
    """Create a tiny ONNX model that exposes a segment-level logits output."""
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["input_logits"], ["logits"], name="IdentityLogits")],
        name="LogitsIdentity",
        inputs=[
            helper.make_tensor_value_info(
                "input_logits",
                TensorProto.FLOAT,
                ["num_segments", 2],
            )
        ],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["num_segments", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    # Keep test models loadable by older ONNX Runtime versions.
    model.ir_version = 10
    return model


def run_augmented_model(logits):
    """Run the augmented graph in ONNX Runtime."""
    model = add_transcript_voting_outputs(make_logits_identity_model())
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(
        [
            "logits",
            SEGMENT_PROBABILITIES,
            SEGMENT_LABELS,
            TRANSCRIPT_PROBABILITIES,
            TRANSCRIPT_LABEL_AVERAGE,
        ],
        {"input_logits": np.asarray(logits, dtype=np.float32)},
    )


def test_transcript_voting_outputs_match_numpy_reference():
    """Average-probability outputs should match NumPy references."""
    logits = np.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )

    outputs = run_augmented_model(logits)
    _, segment_probs, segment_labels, transcript_probs, average_label = outputs

    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected_probs = exp / exp.sum(axis=-1, keepdims=True)
    expected_segment_labels = expected_probs.argmax(axis=-1).astype(np.int64)
    expected_transcript_probs = expected_probs.mean(axis=0)

    assert np.allclose(segment_probs, expected_probs, atol=1e-6)
    assert np.array_equal(segment_labels, expected_segment_labels)
    assert np.allclose(transcript_probs, expected_transcript_probs, atol=1e-6)
    assert int(average_label[0]) == int(expected_transcript_probs.argmax())


def test_average_vote_tie_maps_to_zero():
    """Average probability ties should map to class 0 (numpy argmax behavior)."""
    logits = np.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )

    outputs = run_augmented_model(logits)
    average_label = outputs[-1]

    assert int(average_label[0]) == 0
