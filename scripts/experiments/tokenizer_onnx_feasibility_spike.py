#!/usr/bin/env python3
"""Run tokenizer ONNX feasibility spike with ORT Extensions parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import transformers
from onnx import save
from onnxruntime_extensions import gen_processing_models, get_library_path
from transformers import AutoTokenizer


EDGE_CASES: List[Tuple[str, str]] = [
    ("empty", ""),
    ("whitespace", "   \t  "),
    ("unicode", "Paciente relata deja vu 心理状态 with emoji 🙂 and accents cliche naive."),
    ("punctuation_heavy", "??? !!! ... ,,, ;; :: -- (([])) \"' @#$%^&*"),
    ("long_text", " ".join(["The patient reports unusual thoughts and confusion."] * 800)),
    (
        "clinical_like",
        "Interviewer asks about suspiciousness and unusual perceptual experiences over the last week.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenizer ONNX feasibility spike for String-In ONNX checklist section 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/string_onnx_checkpoints"),
        help="Root with per-backbone best_model directories",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default="bert,clinicalbert,mentalbert",
        help="Comma-separated backbone names",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/tokenizer_spike"),
        help="Output root for tokenizer ONNX models",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/tokenizer_spike"),
        help="Output root for spike reports",
    )
    parser.add_argument(
        "--ortx-library-path",
        type=Path,
        help="Optional explicit ORT Extensions shared library path",
    )
    return parser.parse_args()


def first_diff(a: np.ndarray, b: np.ndarray) -> Dict:
    if a.shape != b.shape:
        return {
            "reason": "shape_mismatch",
            "onnx_shape": list(a.shape),
            "hf_shape": list(b.shape),
        }

    diff_positions = np.where(a != b)[0]
    if diff_positions.size == 0:
        return {}

    idx = int(diff_positions[0])
    return {
        "reason": "value_mismatch",
        "index": idx,
        "onnx_value": int(a[idx]),
        "hf_value": int(b[idx]),
    }


def hf_expected_arrays(tokenizer, text: str) -> Dict[str, np.ndarray]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"].astype(np.int64).reshape(-1)
    attention_mask = encoded["attention_mask"].astype(np.int64).reshape(-1)
    token_type_ids = (
        encoded["token_type_ids"].astype(np.int64).reshape(-1)
        if "token_type_ids" in encoded
        else np.zeros_like(input_ids, dtype=np.int64)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def run_backbone(
    backbone: str,
    checkpoint_root: Path,
    output_root: Path,
    report_root: Path,
    ortx_library_path: Path,
) -> Dict:
    model_dir = checkpoint_root / backbone / "best_model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Backbone model directory not found: {model_dir}")

    model_output_dir = output_root / backbone / "1"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_output_dir / "model.onnx"

    report: Dict = {
        "backbone": backbone,
        "model_dir": str(model_dir),
        "onnx_model_path": str(model_path),
        "ortx_library_path": str(ortx_library_path),
        "versions": {
            "transformers": transformers.__version__,
            "onnxruntime": ort.__version__,
        },
    }

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pre_model, _ = gen_processing_models(tokenizer, pre_kwargs={})
    save(pre_model, str(model_path))

    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    report["onnx_checker_passed"] = True

    ort_without_custom_ops = {
        "load_passed": False,
        "error_type": None,
        "error_message": None,
    }
    try:
        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        ort_without_custom_ops["load_passed"] = True
    except Exception as exc:  # noqa: BLE001
        ort_without_custom_ops["error_type"] = type(exc).__name__
        ort_without_custom_ops["error_message"] = str(exc)
    report["ort_without_custom_ops"] = ort_without_custom_ops

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(str(ortx_library_path))
    session = ort.InferenceSession(
        str(model_path),
        session_options,
        providers=["CPUExecutionProvider"],
    )
    report["ort_with_custom_ops_loaded"] = True

    output_names = [item.name for item in session.get_outputs()]
    report["onnx_outputs"] = output_names

    expected_names = ["input_ids", "attention_mask", "token_type_ids"]
    missing_outputs = [name for name in expected_names if name not in output_names]
    if missing_outputs:
        report["status"] = "failed"
        report["reason"] = "missing_expected_outputs"
        report["missing_outputs"] = missing_outputs
        report_path = report_root / f"{backbone}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        return report

    case_results = []
    parity_exact = True

    for case_name, text in EDGE_CASES:
        ort_payload = {"text": np.array([text], dtype=object)}
        ort_result_values = session.run(None, ort_payload)
        ort_result = {
            name: np.asarray(value).astype(np.int64).reshape(-1)
            for name, value in zip(output_names, ort_result_values)
        }

        hf_expected = hf_expected_arrays(tokenizer, text)
        tensor_results = {}

        for tensor_name in expected_names:
            onnx_array = ort_result[tensor_name]
            hf_array = hf_expected[tensor_name]
            equal = bool(np.array_equal(onnx_array, hf_array))
            if not equal:
                parity_exact = False

            tensor_results[tensor_name] = {
                "exact_match": equal,
                "onnx_length": int(onnx_array.shape[0]),
                "hf_length": int(hf_array.shape[0]),
                "first_difference": first_diff(onnx_array, hf_array) if not equal else {},
            }

        case_results.append(
            {
                "case": case_name,
                "input_chars": len(text),
                "all_tensors_exact": all(
                    tensor_results[name]["exact_match"] for name in expected_names
                ),
                "tensors": tensor_results,
            }
        )

    report["edge_cases"] = case_results
    report["parity_exact"] = parity_exact
    report["status"] = "passed" if parity_exact else "failed"

    report_path = report_root / f"{backbone}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parse_args()
    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    if not backbones:
        raise SystemExit("No backbones provided")

    if args.ortx_library_path:
        ortx_library_path = args.ortx_library_path
    else:
        ortx_library_path = Path(get_library_path())

    if not ortx_library_path.exists():
        raise SystemExit(f"ORT Extensions library path not found: {ortx_library_path}")

    reports = []
    for backbone in backbones:
        report = run_backbone(
            backbone=backbone,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            report_root=args.report_root,
            ortx_library_path=ortx_library_path,
        )
        reports.append(report)

    summary = {
        "status": "passed" if all(item["status"] == "passed" for item in reports) else "failed",
        "backbones": [
            {
                "backbone": item["backbone"],
                "status": item["status"],
                "parity_exact": item.get("parity_exact", False),
                "ort_with_custom_ops_loaded": item.get("ort_with_custom_ops_loaded", False),
                "ort_without_custom_ops_load_passed": item.get("ort_without_custom_ops", {}).get(
                    "load_passed", False
                ),
            }
            for item in reports
        ],
    }
    print(json.dumps(summary, indent=2))

    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
