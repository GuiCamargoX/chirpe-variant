#!/usr/bin/env python3
"""Compare Hugging Face and ONNX Runtime logits for parity checks."""

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'onnxruntime'. Install with: conda run -n clinicagent pip install onnxruntime"
    ) from exc


DEFAULT_TEXTS = [
    "I have been feeling mostly normal lately and sleeping well.",
    "Sometimes I hear faint whispers when nobody is nearby.",
    "My concentration has been worse during class this month.",
    "I do not think anything unusual is happening to me.",
    "At times my thoughts race and I cannot stay focused.",
    "I feel suspicious that strangers are watching me.",
    "My mood has been stable and daily routine is unchanged.",
    "I occasionally feel detached from my surroundings.",
    "Loud places make me anxious and confused.",
    "I can still complete my tasks without major problems.",
    "Sometimes I worry people can read my mind.",
    "I rarely notice anything odd in my perception.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify logit parity between HF model and ONNX Runtime model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        required=True,
        help="Directory containing a Hugging Face model and tokenizer",
    )
    parser.add_argument(
        "--onnx-model-path",
        type=Path,
        required=True,
        help="Path to exported ONNX model",
    )
    parser.add_argument(
        "--texts-file",
        type=Path,
        help="Optional JSON file with a list of strings",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of text samples to evaluate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer max length",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for np.allclose",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for np.allclose",
    )
    parser.add_argument(
        "--allow-prediction-mismatch",
        action="store_true",
        help="Do not fail if argmax labels differ",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help="Optional path to write JSON report",
    )
    return parser.parse_args()


def load_texts(texts_file: Path, num_samples: int) -> List[str]:
    if texts_file:
        with open(texts_file, "r") as file:
            payload = json.load(file)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("--texts-file must contain a JSON list of strings")
        source = payload
    else:
        source = DEFAULT_TEXTS

    if not source:
        raise ValueError("No input texts available for parity check")

    if num_samples <= len(source):
        return source[:num_samples]

    samples: List[str] = []
    index = 0
    while len(samples) < num_samples:
        samples.append(source[index % len(source)])
        index += 1
    return samples


def build_ort_inputs(encoded_batch: Dict[str, torch.Tensor], input_names: List[str]) -> Dict[str, np.ndarray]:
    if "input_ids" not in encoded_batch:
        raise ValueError("Tokenized batch does not include input_ids")

    ort_inputs: Dict[str, np.ndarray] = {}
    input_ids = encoded_batch["input_ids"]

    for name in input_names:
        if name in encoded_batch:
            ort_inputs[name] = encoded_batch[name].cpu().numpy()
            continue

        if name == "token_type_ids":
            ort_inputs[name] = torch.zeros_like(input_ids).cpu().numpy()
            continue

        if name == "attention_mask":
            ort_inputs[name] = torch.ones_like(input_ids).cpu().numpy()
            continue

        raise ValueError(f"Cannot build ONNX input tensor for required input '{name}'")

    return ort_inputs


def filter_model_inputs(model, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    accepted_names = set(inspect.signature(model.forward).parameters)
    filtered = {name: tensor for name, tensor in encoded_batch.items() if name in accepted_names}

    if "input_ids" not in filtered:
        raise ValueError("Filtered model inputs do not include input_ids")

    return filtered


def main() -> None:
    args = parse_args()

    if not args.hf_model_dir.exists():
        raise SystemExit(f"HF model directory does not exist: {args.hf_model_dir}")
    if not args.onnx_model_path.exists():
        raise SystemExit(f"ONNX model does not exist: {args.onnx_model_path}")
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be > 0")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be > 0")

    texts = load_texts(args.texts_file, args.num_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.hf_model_dir,
            attn_implementation="eager",
        )
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_dir)
    model.eval()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    model_inputs = filter_model_inputs(model, encoded)

    with torch.no_grad():
        torch_logits = model(**model_inputs).logits.cpu().numpy()

    session = ort.InferenceSession(str(args.onnx_model_path), providers=["CPUExecutionProvider"])
    ort_input_names = [item.name for item in session.get_inputs()]
    ort_output_names = [item.name for item in session.get_outputs()]

    ort_inputs = build_ort_inputs(encoded, ort_input_names)
    ort_outputs = session.run(ort_output_names, ort_inputs)
    ort_logits = ort_outputs[0]

    if torch_logits.shape != ort_logits.shape:
        report = {
            "status": "failed",
            "reason": "shape_mismatch",
            "torch_shape": list(torch_logits.shape),
            "onnx_shape": list(ort_logits.shape),
            "onnx_inputs": ort_input_names,
            "onnx_outputs": ort_output_names,
        }
        print(json.dumps(report, indent=2))
        if args.report_file:
            with open(args.report_file, "w") as file:
                json.dump(report, file, indent=2)
        raise SystemExit(1)

    abs_diff = np.abs(torch_logits - ort_logits)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())

    torch_preds = np.argmax(torch_logits, axis=-1)
    ort_preds = np.argmax(ort_logits, axis=-1)
    prediction_match_rate = float((torch_preds == ort_preds).mean())

    allclose = bool(np.allclose(torch_logits, ort_logits, atol=args.atol, rtol=args.rtol))
    predictions_match = bool(prediction_match_rate == 1.0)

    passed = allclose and (args.allow_prediction_mismatch or predictions_match)

    report = {
        "status": "passed" if passed else "failed",
        "allclose": allclose,
        "predictions_match": predictions_match,
        "prediction_match_rate": prediction_match_rate,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "atol": args.atol,
        "rtol": args.rtol,
        "num_samples": len(texts),
        "torch_shape": list(torch_logits.shape),
        "onnx_shape": list(ort_logits.shape),
        "onnx_inputs": ort_input_names,
        "onnx_outputs": ort_output_names,
    }

    print(json.dumps(report, indent=2))

    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_file, "w") as file:
            json.dump(report, file, indent=2)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
