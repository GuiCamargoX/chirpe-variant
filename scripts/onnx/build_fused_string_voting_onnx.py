#!/usr/bin/env python3
"""Build fixed-slot fused string-input ONNX models with transcript voting.

The generated graph accepts one transcript per request as:

    text: tensor(string)[max_segments]
    segment_mask: tensor(float)[max_segments]

Each text slot is tokenized inside ONNX, padded/truncated to a fixed sequence
length, stacked into a segment batch, classified, and then aggregated with
masked transcript-level average-probability voting.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, save
from onnxruntime_extensions import gen_processing_models, get_library_path
from transformers import AutoTokenizer

DEFAULT_BACKBONES = "bert,clinicalbert,mentalbert"
OUTPUT_NAMES = [
    "logits",
    "probabilities",
    "label",
    "transcript_probabilities",
    "transcript_label_average",
]


def public_path(path: Path) -> str:
    """Return a repo-relative path, or a placeholder for external local paths."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"<external:{path.name}>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-slot fused String-In ONNX models with transcript voting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/string_onnx_checkpoints"),
        help="Root directory containing per-backbone best_model checkpoints",
    )
    parser.add_argument(
        "--classifier-repo",
        type=Path,
        default=Path("outputs/string_onnx_baseline_classifier"),
        help="Repository root with classifier-only ONNX exports",
    )
    parser.add_argument(
        "--output-repo",
        type=Path,
        default=Path("outputs/string_onnx_fused_voting"),
        help="Output repository root for fixed-slot fused ONNX models",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/fused_voting_smoke"),
        help="Output directory for ORT smoke reports",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default=DEFAULT_BACKBONES,
        help="Comma-separated backbone names",
    )
    parser.add_argument(
        "--classifier-suffix",
        type=str,
        default="baseline",
        help="Classifier model name suffix (chirpe_<backbone>_<suffix>)",
    )
    parser.add_argument(
        "--fused-suffix",
        type=str,
        default="string_voting",
        help="Fused model name suffix (chirpe_<backbone>_<suffix>)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Model version directory",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=15,
        help="Fixed number of segment string slots per transcript request",
    )
    parser.add_argument(
        "--sample-texts",
        nargs="*",
        default=[
            "The participant reports occasional suspiciousness.",
            "The participant describes confusion and unusual perceptual experiences.",
        ],
        help="Sample active segment texts used for ORT smoke tests",
    )
    parser.add_argument(
        "--ortx-library-path",
        type=Path,
        help="Optional explicit path to ORT Extensions shared library",
    )
    parser.add_argument(
        "--ortx-library-container-path",
        type=str,
        default="/opt/ortx/libortextensions.so",
        help="Container path used in Triton config op_library_filename",
    )
    return parser.parse_args()


def get_default_opset(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    raise ValueError("No default ai.onnx opset import found")


def get_tensor_shape(value_info) -> List[Union[int, str, None]]:
    dims: List[Union[int, str, None]] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def infer_num_labels(classifier_model: onnx.ModelProto, logits_output_name: str = "logits") -> int:
    for output in classifier_model.graph.output:
        if output.name == logits_output_name:
            shape = get_tensor_shape(output)
            if len(shape) >= 2 and isinstance(shape[-1], int):
                return shape[-1]
    return 2


def int64_initializer(name: str, dims: Sequence[int], values: Sequence[int]):
    return helper.make_tensor(name, TensorProto.INT64, list(dims), list(values))


def float_initializer(name: str, dims: Sequence[int], values: Sequence[float]):
    return helper.make_tensor(name, TensorProto.FLOAT, list(dims), list(values))


def merge_opsets(*models: onnx.ModelProto) -> List[onnx.OperatorSetIdProto]:
    versions: Dict[str, int] = {}
    for model in models:
        for opset in model.opset_import:
            domain = opset.domain
            versions[domain] = max(versions.get(domain, 0), int(opset.version))
    return [helper.make_opsetid(domain, version) for domain, version in versions.items()]


def get_tokenizer_node(tokenizer_model: onnx.ModelProto) -> onnx.NodeProto:
    if len(tokenizer_model.graph.node) != 1:
        raise ValueError(
            "Expected ORT Extensions tokenizer graph to contain one tokenizer node, "
            f"found {len(tokenizer_model.graph.node)}"
        )
    return tokenizer_model.graph.node[0]


def tokenizer_output_names(tokenizer_model: onnx.ModelProto) -> List[str]:
    return [item.name for item in tokenizer_model.graph.output]


def load_max_length(checkpoint_dir: Path, fallback: int = 128) -> int:
    config_path = checkpoint_dir / "chirpe_config.json"
    if not config_path.exists():
        return fallback
    payload = json.loads(config_path.read_text())
    max_length = int(payload.get("model", {}).get("max_length", fallback))
    return max(max_length, 1)


def build_slot_inputs(texts: Sequence[str], max_segments: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad/truncate segment strings and build a 0/1 mask for one transcript."""
    active = list(texts[:max_segments])
    padded = active + [""] * (max_segments - len(active))
    mask = [1.0] * len(active) + [0.0] * (max_segments - len(active))
    return np.array(padded, dtype=object), np.array(mask, dtype=np.float32)


def add_tokenizer_slot_nodes(
    nodes: List[onnx.NodeProto],
    initializers: List[onnx.TensorProto],
    tokenizer_node: onnx.NodeProto,
    tokenizer_outputs: Sequence[str],
    classifier_input_names: Sequence[str],
    slot_index: int,
    max_sequence_length: int,
    pad_token_id: int,
    rows_by_input: Dict[str, List[str]],
) -> None:
    prefix = f"slot_{slot_index}"

    starts_name = f"{prefix}_text_starts"
    ends_name = f"{prefix}_text_ends"
    axes_name = f"{prefix}_axes_zero"
    steps_name = f"{prefix}_steps_one"
    initializers.extend(
        [
            int64_initializer(starts_name, [1], [slot_index]),
            int64_initializer(ends_name, [1], [slot_index + 1]),
            int64_initializer(axes_name, [1], [0]),
            int64_initializer(steps_name, [1], [1]),
        ]
    )

    text_slice = f"{prefix}_text"
    nodes.append(
        helper.make_node(
            "Slice",
            ["text", starts_name, ends_name, axes_name, steps_name],
            [text_slice],
            name=f"{prefix}_SliceText",
        )
    )

    slot_token_outputs = {name: f"{prefix}_{name}_raw" for name in tokenizer_outputs}
    copied_tokenizer_node = copy.deepcopy(tokenizer_node)
    copied_tokenizer_node.name = f"{prefix}_{tokenizer_node.op_type}"
    del copied_tokenizer_node.input[:]
    copied_tokenizer_node.input.extend([text_slice])
    del copied_tokenizer_node.output[:]
    copied_tokenizer_node.output.extend([slot_token_outputs[name] for name in tokenizer_outputs])
    nodes.append(copied_tokenizer_node)

    pad_amount_name = f"{prefix}_pad_amount"
    slice_start_name = f"{prefix}_token_slice_start"
    slice_end_name = f"{prefix}_token_slice_end"
    initializers.extend(
        [
            int64_initializer(pad_amount_name, [2], [0, max_sequence_length]),
            int64_initializer(slice_start_name, [1], [0]),
            int64_initializer(slice_end_name, [1], [max_sequence_length]),
        ]
    )

    for input_name in classifier_input_names:
        if input_name not in slot_token_outputs:
            raise ValueError(f"Tokenizer output missing required classifier input: {input_name}")

        pad_value = pad_token_id if input_name == "input_ids" else 0
        pad_value_name = f"{prefix}_{input_name}_pad_value"
        padded_name = f"{prefix}_{input_name}_padded"
        clipped_name = f"{prefix}_{input_name}_fixed"
        row_name = f"{prefix}_{input_name}_row"
        initializers.append(int64_initializer(pad_value_name, [], [pad_value]))

        nodes.append(
            helper.make_node(
                "Pad",
                [slot_token_outputs[input_name], pad_amount_name, pad_value_name],
                [padded_name],
                name=f"{prefix}_{input_name}_Pad",
                mode="constant",
            )
        )
        nodes.append(
            helper.make_node(
                "Slice",
                [padded_name, slice_start_name, slice_end_name, axes_name, steps_name],
                [clipped_name],
                name=f"{prefix}_{input_name}_SliceFixed",
            )
        )
        nodes.append(
            helper.make_node(
                "Unsqueeze",
                [clipped_name, axes_name],
                [row_name],
                name=f"{prefix}_{input_name}_UnsqueezeRow",
            )
        )
        rows_by_input[input_name].append(row_name)


def add_postprocess_nodes(
    nodes: List[onnx.NodeProto],
    initializers: List[onnx.TensorProto],
) -> None:
    axes_zero = "voting_axes_zero"
    axes_one = "voting_axes_one"
    eps_float = "voting_eps_float"
    initializers.extend(
        [
            int64_initializer(axes_zero, [1], [0]),
            int64_initializer(axes_one, [1], [1]),
            float_initializer(eps_float, [], [1e-6]),
        ]
    )

    nodes.extend(
        [
            helper.make_node(
                "Softmax",
                ["logits"],
                ["probabilities"],
                name="VotingSegmentSoftmax",
                axis=-1,
            ),
            helper.make_node(
                "ArgMax",
                ["probabilities"],
                ["label"],
                name="VotingSegmentLabel",
                axis=-1,
                keepdims=0,
            ),
            helper.make_node(
                "Unsqueeze",
                ["segment_mask", axes_one],
                ["segment_mask_column"],
                name="VotingMaskColumn",
            ),
            helper.make_node(
                "Mul",
                ["probabilities", "segment_mask_column"],
                ["masked_probabilities"],
                name="VotingMaskedProbabilities",
            ),
            helper.make_node(
                "ReduceSum",
                ["masked_probabilities", axes_zero],
                ["masked_probability_sum"],
                name="VotingProbabilitySum",
                keepdims=0,
            ),
            helper.make_node(
                "ReduceSum",
                ["segment_mask", axes_zero],
                ["segment_mask_sum"],
                name="VotingMaskSum",
                keepdims=0,
            ),
            helper.make_node(
                "Max",
                ["segment_mask_sum", eps_float],
                ["safe_segment_mask_sum"],
                name="VotingSafeMaskSum",
            ),
            helper.make_node(
                "Div",
                ["masked_probability_sum", "safe_segment_mask_sum"],
                ["transcript_probabilities"],
                name="VotingTranscriptProbabilities",
            ),
            helper.make_node(
                "ArgMax",
                ["transcript_probabilities"],
                ["transcript_label_average"],
                name="VotingTranscriptAverageLabel",
                axis=0,
                keepdims=1,
            ),
        ]
    )


def build_fused_voting_model(
    tokenizer_model: onnx.ModelProto,
    classifier_model: onnx.ModelProto,
    tokenizer,
    max_segments: int,
    max_sequence_length: int,
) -> onnx.ModelProto:
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")

    opset = get_default_opset(classifier_model)
    if opset < 13:
        raise ValueError("Fixed-slot fused voting model requires classifier ONNX opset >= 13")

    tokenizer_node = get_tokenizer_node(tokenizer_model)
    token_outputs = tokenizer_output_names(tokenizer_model)
    classifier_input_names = [item.name for item in classifier_model.graph.input]
    num_labels = infer_num_labels(classifier_model)
    pad_token_id = int(tokenizer.pad_token_id or 0)

    nodes: List[onnx.NodeProto] = []
    initializers: List[onnx.TensorProto] = []
    rows_by_input: Dict[str, List[str]] = {name: [] for name in classifier_input_names}

    for slot_index in range(max_segments):
        add_tokenizer_slot_nodes(
            nodes=nodes,
            initializers=initializers,
            tokenizer_node=tokenizer_node,
            tokenizer_outputs=token_outputs,
            classifier_input_names=classifier_input_names,
            slot_index=slot_index,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
            rows_by_input=rows_by_input,
        )

    for input_name in classifier_input_names:
        nodes.append(
            helper.make_node(
                "Concat",
                rows_by_input[input_name],
                [input_name],
                name=f"Concat_{input_name}",
                axis=0,
            )
        )

    nodes.extend(copy.deepcopy(classifier_model.graph.node))
    initializers.extend(copy.deepcopy(classifier_model.graph.initializer))
    add_postprocess_nodes(nodes, initializers)

    graph = helper.make_graph(
        nodes=nodes,
        name="ChirpeFixedSlotStringVoting",
        inputs=[
            helper.make_tensor_value_info("text", TensorProto.STRING, [max_segments]),
            helper.make_tensor_value_info("segment_mask", TensorProto.FLOAT, [max_segments]),
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [max_segments, num_labels]),
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [max_segments, num_labels]
            ),
            helper.make_tensor_value_info("label", TensorProto.INT64, [max_segments]),
            helper.make_tensor_value_info(
                "transcript_probabilities", TensorProto.FLOAT, [num_labels]
            ),
            helper.make_tensor_value_info("transcript_label_average", TensorProto.INT64, [1]),
        ],
        initializer=initializers,
        value_info=copy.deepcopy(classifier_model.graph.value_info),
    )

    model = helper.make_model(
        graph,
        opset_imports=merge_opsets(classifier_model, tokenizer_model),
        producer_name="chirpe-fixed-slot-string-voting",
    )
    model.ir_version = classifier_model.ir_version
    onnx.checker.check_model(model)
    return model


def infer_metadata(session: ort.InferenceSession) -> Dict:
    return {
        "inputs": [
            {"name": item.name, "type": item.type, "shape": list(item.shape)}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "type": item.type, "shape": list(item.shape)}
            for item in session.get_outputs()
        ],
    }


def numpy_voting_reference(logits: np.ndarray, segment_mask: np.ndarray) -> Dict:
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    labels = probabilities.argmax(axis=-1).astype(np.int64)
    safe_count = max(float(segment_mask.sum()), 1e-6)
    transcript_probabilities = (probabilities * segment_mask[:, None]).sum(axis=0) / safe_count
    transcript_label_average = int(transcript_probabilities.argmax())
    return {
        "probabilities": probabilities,
        "labels": labels,
        "transcript_probabilities": transcript_probabilities,
        "transcript_label_average": transcript_label_average,
    }


def format_dims(dims: Sequence[Union[int, str, None]]) -> str:
    formatted = []
    for dim in dims:
        formatted.append(str(dim) if isinstance(dim, int) else "-1")
    return ", ".join(formatted)


def triton_dtype(ort_type: str) -> str:
    mapping = {
        "tensor(float)": "TYPE_FP32",
        "tensor(int64)": "TYPE_INT64",
        "tensor(string)": "TYPE_STRING",
    }
    if ort_type not in mapping:
        raise ValueError(f"Unsupported ORT type for generated Triton config: {ort_type}")
    return mapping[ort_type]


def build_config_pbtxt(
    model_name: str,
    session: ort.InferenceSession,
    ortx_library_container_path: str,
) -> str:
    input_lines = []
    for index, item in enumerate(session.get_inputs()):
        suffix = "," if index < len(session.get_inputs()) - 1 else ""
        input_lines.append(
            f'  {{ name: "{item.name}" data_type: {triton_dtype(item.type)} dims: [ {format_dims(item.shape)} ] }}{suffix}'
        )

    output_lines = []
    for index, item in enumerate(session.get_outputs()):
        suffix = "," if index < len(session.get_outputs()) - 1 else ""
        output_lines.append(
            f'  {{ name: "{item.name}" data_type: {triton_dtype(item.type)} dims: [ {format_dims(item.shape)} ] }}{suffix}'
        )

    return "\n".join(
        [
            f'name: "{model_name}"',
            'backend: "onnxruntime"',
            "max_batch_size: 0",
            "",
            "input [",
            *input_lines,
            "]",
            "",
            "output [",
            *output_lines,
            "]",
            "",
            "model_operations: {",
            f'  op_library_filename: ["{ortx_library_container_path}"]',
            "}",
            "",
            "instance_group [",
            "  { kind: KIND_CPU }",
            "]",
            "",
        ]
    )


def run_backbone(
    backbone: str,
    checkpoint_root: Path,
    classifier_repo: Path,
    output_repo: Path,
    report_root: Path,
    classifier_suffix: str,
    fused_suffix: str,
    version: int,
    max_segments: int,
    sample_texts: Sequence[str],
    ortx_library_path: Path,
    ortx_library_container_path: str,
) -> Dict:
    checkpoint_dir = checkpoint_root / backbone / "best_model"
    classifier_model_name = f"chirpe_{backbone}_{classifier_suffix}"
    fused_model_name = f"chirpe_{backbone}_{fused_suffix}"
    classifier_onnx_path = classifier_repo / classifier_model_name / str(version) / "model.onnx"
    fused_root = output_repo / fused_model_name
    fused_version_dir = fused_root / str(version)
    fused_version_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{backbone}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fused_model_path = fused_version_dir / "model.onnx"
    metadata_path = fused_version_dir / "export_metadata.json"
    config_path = fused_root / "config.pbtxt"

    report: Dict = {
        "backbone": backbone,
        "status": "failed",
        "checkpoint_dir": str(checkpoint_dir),
        "classifier_onnx_path": str(classifier_onnx_path),
        "fused_model_name": fused_model_name,
        "fused_model_path": str(fused_model_path),
        "max_segments": max_segments,
    }

    if not checkpoint_dir.exists():
        report["reason"] = "checkpoint_missing"
        report_path.write_text(json.dumps(report, indent=2))
        return report
    if not classifier_onnx_path.exists():
        report["reason"] = "classifier_onnx_missing"
        report_path.write_text(json.dumps(report, indent=2))
        return report

    max_sequence_length = load_max_length(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer_model, _ = gen_processing_models(tokenizer, pre_kwargs={})
    classifier_model = onnx.load(str(classifier_onnx_path))
    tokenizer_model.ir_version = classifier_model.ir_version

    model = build_fused_voting_model(
        tokenizer_model=tokenizer_model,
        classifier_model=classifier_model,
        tokenizer=tokenizer,
        max_segments=max_segments,
        max_sequence_length=max_sequence_length,
    )
    save(model, str(fused_model_path))
    onnx.checker.check_model(str(fused_model_path))
    report["onnx_checker_passed"] = True

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(str(ortx_library_path))
    session = ort.InferenceSession(
        str(fused_model_path),
        session_options,
        providers=["CPUExecutionProvider"],
    )

    texts, segment_mask = build_slot_inputs(sample_texts, max_segments=max_segments)
    outputs = session.run(OUTPUT_NAMES, {"text": texts, "segment_mask": segment_mask})
    logits, probabilities, labels, transcript_probs, average_label = outputs
    reference = numpy_voting_reference(logits, segment_mask)

    smoke_valid = bool(
        logits.shape == (max_segments, 2)
        and probabilities.shape == logits.shape
        and labels.shape == (max_segments,)
        and transcript_probs.shape == (2,)
        and average_label.shape == (1,)
        and np.allclose(probabilities, reference["probabilities"], atol=1e-6)
        and np.array_equal(labels, reference["labels"])
        and np.allclose(transcript_probs, reference["transcript_probabilities"], atol=1e-6)
        and int(average_label[0]) == reference["transcript_label_average"]
    )

    io_metadata = infer_metadata(session)
    metadata = {
        "backbone": backbone,
        "fused_model_name": fused_model_name,
        "source_checkpoint_dir": public_path(checkpoint_dir),
        "source_classifier_onnx_path": public_path(classifier_onnx_path),
        "ortx_library_required": public_path(ortx_library_path),
        "max_segments": max_segments,
        "max_sequence_length": max_sequence_length,
        "contract": {
            "text": f"tensor(string)[{max_segments}]",
            "segment_mask": f"tensor(float)[{max_segments}], with 1 for active slots and 0 for padded slots",
        },
        "inputs": io_metadata["inputs"],
        "outputs": io_metadata["outputs"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    config_path.write_text(
        build_config_pbtxt(
            model_name=fused_model_name,
            session=session,
            ortx_library_container_path=ortx_library_container_path,
        )
    )

    report.update(
        {
            "status": "passed" if smoke_valid else "failed",
            "smoke_inference_valid": smoke_valid,
            "max_sequence_length": max_sequence_length,
            "active_sample_segments": int(segment_mask.sum()),
            "logits_shape": list(logits.shape),
            "probabilities_shape": list(probabilities.shape),
            "label_shape": list(labels.shape),
            "transcript_probabilities_shape": list(transcript_probs.shape),
            "transcript_label_average": int(average_label[0]),
            "metadata_path": str(metadata_path),
            "config_path": str(config_path),
            "inputs": io_metadata["inputs"],
            "outputs": io_metadata["outputs"],
        }
    )
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parse_args()
    if args.version <= 0:
        raise SystemExit("--version must be a positive integer")
    if args.max_segments <= 0:
        raise SystemExit("--max-segments must be positive")

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    if not backbones:
        raise SystemExit("No backbones provided")

    ortx_library_path = (
        args.ortx_library_path if args.ortx_library_path else Path(get_library_path())
    )
    if not ortx_library_path.exists():
        raise SystemExit(f"ORT Extensions library path not found: {ortx_library_path}")

    reports = []
    for backbone in backbones:
        reports.append(
            run_backbone(
                backbone=backbone,
                checkpoint_root=args.checkpoint_root,
                classifier_repo=args.classifier_repo,
                output_repo=args.output_repo,
                report_root=args.report_root,
                classifier_suffix=args.classifier_suffix,
                fused_suffix=args.fused_suffix,
                version=args.version,
                max_segments=args.max_segments,
                sample_texts=args.sample_texts,
                ortx_library_path=ortx_library_path,
                ortx_library_container_path=args.ortx_library_container_path,
            )
        )

    summary = {
        "status": "passed" if all(item.get("status") == "passed" for item in reports) else "failed",
        "reports": [
            {
                "backbone": item.get("backbone"),
                "status": item.get("status"),
                "fused_model_path": item.get("fused_model_path"),
                "metadata_path": item.get("metadata_path"),
                "config_path": item.get("config_path"),
                "smoke_inference_valid": item.get("smoke_inference_valid", False),
            }
            for item in reports
        ],
    }
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
