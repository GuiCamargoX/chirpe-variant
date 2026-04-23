#!/usr/bin/env python3
"""Build fused tokenizer+classifier ONNX models (string -> logits)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.compose
import onnxruntime as ort
from onnx import TensorProto, helper, save
from onnxruntime_extensions import gen_processing_models, get_library_path
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fused String-In ONNX models (tokenizer + classifier)",
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
        default=Path("outputs/string_onnx_fused"),
        help="Output repository root for fused ONNX models",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/fused_smoke"),
        help="Output directory for fused ORT smoke reports",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default="bert,clinicalbert,mentalbert",
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
        default="string",
        help="Fused model name suffix (chirpe_<backbone>_<suffix>)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Model version directory",
    )
    parser.add_argument(
        "--sample-text",
        type=str,
        default="The participant reports occasional suspiciousness and confusion.",
        help="Sample text used for fused ORT inference smoke test",
    )
    parser.add_argument(
        "--ortx-library-path",
        type=Path,
        help="Optional explicit path to ORT Extensions shared library",
    )
    return parser.parse_args()


def get_opset_version(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    raise ValueError("No default ai.onnx opset import found")


def build_adapter_model(
    classifier_input_names: List[str],
    opset_version: int,
    ir_version: int,
    max_sequence_length: int,
) -> Tuple[onnx.ModelProto, List[Tuple[str, str]], List[Tuple[str, str]]]:
    inputs = []
    outputs = []
    nodes = []
    io_map_tok_to_adapter: List[Tuple[str, str]] = []
    io_map_adapter_to_classifier: List[Tuple[str, str]] = []

    axes_init = helper.make_tensor("axes_zero", TensorProto.INT64, [1], [0])
    starts_init = helper.make_tensor("slice_starts", TensorProto.INT64, [1], [0])
    ends_init = helper.make_tensor("slice_ends", TensorProto.INT64, [1], [max_sequence_length])
    slice_axes_init = helper.make_tensor("slice_axes", TensorProto.INT64, [1], [0])
    slice_steps_init = helper.make_tensor("slice_steps", TensorProto.INT64, [1], [1])

    for name in classifier_input_names:
        adapter_input = f"tok_{name}"
        clipped_output = f"tok_{name}_clipped"
        adapter_output = f"cls_{name}"

        inputs.append(
            helper.make_tensor_value_info(adapter_input, TensorProto.INT64, ["seq_len"])
        )
        outputs.append(
            helper.make_tensor_value_info(
                adapter_output,
                TensorProto.INT64,
                ["batch_size", "sequence_length"],
            )
        )
        nodes.append(
            helper.make_node(
                "Slice",
                [adapter_input, "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
                [clipped_output],
                name=f"Slice_{name}",
            )
        )
        nodes.append(
            helper.make_node(
                "Unsqueeze",
                [clipped_output, "axes_zero"],
                [adapter_output],
                name=f"Unsqueeze_{name}",
            )
        )

        io_map_tok_to_adapter.append((name, adapter_input))
        io_map_adapter_to_classifier.append((adapter_output, name))

    graph = helper.make_graph(
        nodes,
        "TokenizerClassifierAdapter",
        inputs,
        outputs,
        [axes_init, starts_init, ends_init, slice_axes_init, slice_steps_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
    model.ir_version = ir_version
    return model, io_map_tok_to_adapter, io_map_adapter_to_classifier


def infer_metadata(session: ort.InferenceSession) -> Dict:
    return {
        "inputs": [
            {
                "name": item.name,
                "type": item.type,
                "shape": list(item.shape),
            }
            for item in session.get_inputs()
        ],
        "outputs": [
            {
                "name": item.name,
                "type": item.type,
                "shape": list(item.shape),
            }
            for item in session.get_outputs()
        ],
    }


def run_backbone(
    backbone: str,
    checkpoint_root: Path,
    classifier_repo: Path,
    output_repo: Path,
    report_root: Path,
    classifier_suffix: str,
    fused_suffix: str,
    version: int,
    sample_text: str,
    ortx_library_path: Path,
) -> Dict:
    checkpoint_dir = checkpoint_root / backbone / "best_model"
    classifier_model_name = f"chirpe_{backbone}_{classifier_suffix}"
    fused_model_name = f"chirpe_{backbone}_{fused_suffix}"

    classifier_onnx_path = classifier_repo / classifier_model_name / str(version) / "model.onnx"
    fused_version_dir = output_repo / fused_model_name / str(version)
    fused_version_dir.mkdir(parents=True, exist_ok=True)
    fused_model_path = fused_version_dir / "model.onnx"
    fused_metadata_path = fused_version_dir / "export_metadata.json"
    report_path = report_root / f"{backbone}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict = {
        "backbone": backbone,
        "status": "failed",
        "checkpoint_dir": str(checkpoint_dir),
        "classifier_onnx_path": str(classifier_onnx_path),
        "fused_model_name": fused_model_name,
        "fused_model_path": str(fused_model_path),
        "ortx_library_path": str(ortx_library_path),
    }

    if not checkpoint_dir.exists():
        report["reason"] = "checkpoint_missing"
        report_path.write_text(json.dumps(report, indent=2))
        return report
    if not classifier_onnx_path.exists():
        report["reason"] = "classifier_onnx_missing"
        report_path.write_text(json.dumps(report, indent=2))
        return report

    config_path = checkpoint_dir / "chirpe_config.json"
    max_sequence_length = 128
    if config_path.exists():
        config_payload = json.loads(config_path.read_text())
        max_sequence_length = int(config_payload.get("model", {}).get("max_length", max_sequence_length))
    if max_sequence_length <= 0:
        max_sequence_length = 128
    report["max_sequence_length"] = max_sequence_length

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer_model, _ = gen_processing_models(tokenizer, pre_kwargs={})
    classifier_model = onnx.load(str(classifier_onnx_path))

    # onnx.compose.merge_models requires same IR version across models.
    tokenizer_model.ir_version = classifier_model.ir_version
    opset_version = get_opset_version(classifier_model)

    classifier_input_names = [item.name for item in classifier_model.graph.input]
    required_tokenizer_outputs = set(classifier_input_names)
    tokenizer_output_names = {item.name for item in tokenizer_model.graph.output}
    missing_outputs = sorted(required_tokenizer_outputs - tokenizer_output_names)
    if missing_outputs:
        report["reason"] = "tokenizer_missing_required_outputs"
        report["missing_outputs"] = missing_outputs
        report_path.write_text(json.dumps(report, indent=2))
        return report

    adapter_model, io_map_tok_to_adapter, io_map_adapter_to_classifier = build_adapter_model(
        classifier_input_names=classifier_input_names,
        opset_version=opset_version,
        ir_version=classifier_model.ir_version,
        max_sequence_length=max_sequence_length,
    )

    merged_tokenizer_adapter = onnx.compose.merge_models(
        tokenizer_model,
        adapter_model,
        io_map=io_map_tok_to_adapter,
        inputs=["text"],
        outputs=[pair[0] for pair in io_map_adapter_to_classifier],
        name=f"tokenizer_adapter_{backbone}",
    )
    merged_tokenizer_adapter.ir_version = classifier_model.ir_version

    fused_model = onnx.compose.merge_models(
        merged_tokenizer_adapter,
        classifier_model,
        io_map=io_map_adapter_to_classifier,
        inputs=["text"],
        outputs=[item.name for item in classifier_model.graph.output],
        name=f"fused_string_classifier_{backbone}",
    )
    fused_model.ir_version = classifier_model.ir_version

    save(fused_model, str(fused_model_path))
    onnx.checker.check_model(fused_model)
    report["onnx_checker_passed"] = True

    without_custom_ops = {"load_passed": False, "error_type": None, "error_message": None}
    try:
        ort.InferenceSession(str(fused_model_path), providers=["CPUExecutionProvider"])
        without_custom_ops["load_passed"] = True
    except Exception as exc:  # noqa: BLE001
        without_custom_ops["error_type"] = type(exc).__name__
        without_custom_ops["error_message"] = str(exc)
    report["ort_without_custom_ops"] = without_custom_ops

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(str(ortx_library_path))
    session = ort.InferenceSession(
        str(fused_model_path),
        session_options,
        providers=["CPUExecutionProvider"],
    )
    report["ort_with_custom_ops_loaded"] = True

    smoke_logits = session.run(None, {"text": np.array([sample_text], dtype=object)})[0]
    smoke_valid = bool(
        isinstance(smoke_logits, np.ndarray)
        and smoke_logits.ndim == 2
        and smoke_logits.shape[0] >= 1
        and np.isfinite(smoke_logits).all()
    )

    io_metadata = infer_metadata(session)
    metadata_payload = {
        "backbone": backbone,
        "fused_model_name": fused_model_name,
        "source_checkpoint_dir": str(checkpoint_dir),
        "source_classifier_onnx_path": str(classifier_onnx_path),
        "ortx_library_required": str(ortx_library_path),
        "inputs": io_metadata["inputs"],
        "outputs": io_metadata["outputs"],
    }
    fused_metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    report.update(
        {
            "smoke_inference_valid": smoke_valid,
            "smoke_logits_shape": list(smoke_logits.shape),
            "fused_metadata_path": str(fused_metadata_path),
            "inputs": io_metadata["inputs"],
            "outputs": io_metadata["outputs"],
            "status": "passed" if smoke_valid else "failed",
        }
    )

    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parse_args()

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    if not backbones:
        raise SystemExit("No backbones provided")

    if args.version <= 0:
        raise SystemExit("--version must be a positive integer")

    ortx_library_path = args.ortx_library_path if args.ortx_library_path else Path(get_library_path())
    if not ortx_library_path.exists():
        raise SystemExit(f"ORT Extensions library path not found: {ortx_library_path}")

    reports = []
    for backbone in backbones:
        report = run_backbone(
            backbone=backbone,
            checkpoint_root=args.checkpoint_root,
            classifier_repo=args.classifier_repo,
            output_repo=args.output_repo,
            report_root=args.report_root,
            classifier_suffix=args.classifier_suffix,
            fused_suffix=args.fused_suffix,
            version=args.version,
            sample_text=args.sample_text,
            ortx_library_path=ortx_library_path,
        )
        reports.append(report)

    summary = {
        "status": "passed" if all(item.get("status") == "passed" for item in reports) else "failed",
        "reports": [
            {
                "backbone": item.get("backbone"),
                "status": item.get("status"),
                "fused_model_path": item.get("fused_model_path"),
                "fused_metadata_path": item.get("fused_metadata_path"),
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
