#!/usr/bin/env python3
"""Add transcript-level voting outputs to a segment-batch classifier ONNX model.

The input model is expected to emit segment-level logits with shape
`[num_segments, num_labels]`. The generated model keeps existing outputs and
adds ONNX-native post-processing outputs for per-segment probabilities/labels
and transcript-level average-probability aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import onnx
from onnx import TensorProto, helper

SEGMENT_PROBABILITIES = "segment_probabilities"
SEGMENT_LABELS = "segment_labels"
TRANSCRIPT_PROBABILITIES = "transcript_probabilities"
TRANSCRIPT_LABEL_AVERAGE = "transcript_label_average"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Append transcript-level voting outputs to a classifier ONNX model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-model",
        type=Path,
        required=True,
        help="Path to classifier ONNX model that outputs segment logits",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        required=True,
        help="Path where the voting-augmented ONNX model will be written",
    )
    parser.add_argument(
        "--logits-output-name",
        type=str,
        default="logits",
        help="Name of the segment-level logits tensor in the source ONNX graph",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        help="Optional JSON metadata file describing the added outputs",
    )
    return parser.parse_args()


def get_default_opset(model: onnx.ModelProto) -> int:
    """Return the model's default ONNX opset version."""
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    raise ValueError("No default ai.onnx opset import found")


def get_tensor_shape(value_info) -> List[Union[int, str, None]]:
    """Extract a best-effort tensor shape from ONNX value info."""
    tensor_type = value_info.type.tensor_type
    dims: List[Union[int, str, None]] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def find_graph_output(model: onnx.ModelProto, name: str):
    """Find a graph output by name."""
    for output in model.graph.output:
        if output.name == name:
            return output
    output_names = [output.name for output in model.graph.output]
    raise ValueError(f"Output {name!r} not found. Available outputs: {output_names}")


def ensure_names_available(model: onnx.ModelProto, names: Sequence[str]) -> None:
    """Fail if an added output name already exists in the source graph."""
    graph_names = {value.name for value in model.graph.output}
    graph_names.update(value.name for value in model.graph.value_info)
    graph_names.update(value.name for value in model.graph.input)
    graph_names.update(init.name for init in model.graph.initializer)

    conflicts = sorted(set(names) & graph_names)
    if conflicts:
        raise ValueError(f"Model already contains voting output names: {conflicts}")


def infer_num_labels(model: onnx.ModelProto, logits_output_name: str) -> Union[int, str]:
    """Infer the class dimension from the logits output shape when available."""
    logits_output = find_graph_output(model, logits_output_name)
    shape = get_tensor_shape(logits_output)
    if len(shape) >= 2 and shape[-1] is not None:
        return shape[-1]
    return "num_labels"


def make_initializer(name: str, dims: Sequence[int], values: Sequence[int]):
    """Create an INT64 initializer."""
    return helper.make_tensor(name, TensorProto.INT64, list(dims), list(values))


def add_transcript_voting_outputs(
    model: onnx.ModelProto,
    logits_output_name: str = "logits",
) -> onnx.ModelProto:
    """Append ONNX-native segment and transcript aggregation outputs.

    Average aggregation computes `ArgMax(ReduceMean(Softmax(logits)))`.
    """
    opset = get_default_opset(model)
    if opset < 13:
        raise ValueError("Transcript voting graph requires ONNX opset >= 13")

    find_graph_output(model, logits_output_name)
    added_outputs = [
        SEGMENT_PROBABILITIES,
        SEGMENT_LABELS,
        TRANSCRIPT_PROBABILITIES,
        TRANSCRIPT_LABEL_AVERAGE,
    ]
    ensure_names_available(model, added_outputs)

    num_labels = infer_num_labels(model, logits_output_name)

    axes_zero_name = "chirpe_voting_axes_zero"
    model.graph.initializer.extend(
        [
            make_initializer(axes_zero_name, [1], [0]),
        ]
    )

    model.graph.node.extend(
        [
            helper.make_node(
                "Softmax",
                [logits_output_name],
                [SEGMENT_PROBABILITIES],
                name="ChirpeSegmentSoftmax",
                axis=-1,
            ),
            helper.make_node(
                "ArgMax",
                [SEGMENT_PROBABILITIES],
                [SEGMENT_LABELS],
                name="ChirpeSegmentLabels",
                axis=-1,
                keepdims=0,
            ),
            helper.make_node(
                "ReduceMean",
                [SEGMENT_PROBABILITIES, axes_zero_name],
                [TRANSCRIPT_PROBABILITIES],
                name="ChirpeTranscriptAverageProbabilities",
                keepdims=0,
            ),
            helper.make_node(
                "ArgMax",
                [TRANSCRIPT_PROBABILITIES],
                [TRANSCRIPT_LABEL_AVERAGE],
                name="ChirpeTranscriptAverageLabel",
                axis=0,
                keepdims=1,
            ),
        ]
    )

    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                SEGMENT_PROBABILITIES,
                TensorProto.FLOAT,
                ["num_segments", num_labels],
            ),
            helper.make_tensor_value_info(SEGMENT_LABELS, TensorProto.INT64, ["num_segments"]),
            helper.make_tensor_value_info(
                TRANSCRIPT_PROBABILITIES,
                TensorProto.FLOAT,
                [num_labels],
            ),
            helper.make_tensor_value_info(TRANSCRIPT_LABEL_AVERAGE, TensorProto.INT64, [1]),
        ]
    )

    onnx.checker.check_model(model)
    return model


def metadata_payload(
    input_model: Path,
    output_model: Path,
    logits_output_name: str,
) -> Dict:
    """Build metadata describing the voting contract."""
    return {
        "input_model": str(input_model),
        "output_model": str(output_model),
        "logits_output_name": logits_output_name,
        "added_outputs": {
            SEGMENT_PROBABILITIES: "Softmax over segment logits; shape [num_segments, num_labels].",
            SEGMENT_LABELS: "ArgMax over segment probabilities; shape [num_segments].",
            TRANSCRIPT_PROBABILITIES: "Mean segment probabilities; shape [num_labels].",
            TRANSCRIPT_LABEL_AVERAGE: "ArgMax over transcript_probabilities; shape [1].",
        },
        "aggregation_scope": "One ONNX invocation represents one transcript's segment batch.",
    }


def main() -> None:
    """Run voting graph augmentation."""
    args = parse_args()
    if not args.input_model.exists():
        raise SystemExit(f"Input ONNX model does not exist: {args.input_model}")

    model = onnx.load(str(args.input_model))
    model = add_transcript_voting_outputs(
        model=model,
        logits_output_name=args.logits_output_name,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output_model))

    metadata_path = args.metadata_file
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as file:
            json.dump(
                metadata_payload(
                    input_model=args.input_model,
                    output_model=args.output_model,
                    logits_output_name=args.logits_output_name,
                ),
                file,
                indent=2,
            )

    print(f"Voting-augmented ONNX model written: {args.output_model}")
    if metadata_path:
        print(f"Metadata written: {metadata_path}")


if __name__ == "__main__":
    main()
