#!/usr/bin/env python3
"""Compare fused Triton string-input models against Python reference pipeline."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from chirpe.data.preprocessor import TranscriptPreprocessor


BACKBONE_TO_MODEL = {
    "bert": "chirpe_bert_string",
    "clinicalbert": "chirpe_clinicalbert_string",
    "mentalbert": "chirpe_mentalbert_string",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate parity between Triton fused models and Python reference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path("outputs/string_onnx_fused"),
        help="Triton model repository with fused models",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("outputs/string_onnx_checkpoints"),
        help="Root directory containing per-backbone best_model checkpoints",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/synthetic/test.json"),
        help="Transcript JSON file used for parity testing",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default="bert,clinicalbert,mentalbert",
        help="Comma-separated backbone names",
    )
    parser.add_argument(
        "--triton-image",
        type=str,
        default="nvcr.io/nvidia/tritonserver:24.10-py3",
        help="Triton docker image",
    )
    parser.add_argument("--http-port", type=int, default=8000, help="Triton HTTP port")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Readiness timeout")
    parser.add_argument(
        "--max-transcripts",
        type=int,
        default=0,
        help="Maximum number of transcripts to evaluate (0 means all)",
    )
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument(
        "--target-label-agreement",
        type=float,
        default=0.995,
        help="Minimum segment-level label agreement",
    )
    parser.add_argument(
        "--target-transcript-agreement",
        type=float,
        default=0.995,
        help="Minimum transcript-level voting agreement",
    )
    parser.add_argument(
        "--ortx-library-host-path",
        type=Path,
        default=Path("outputs/onnxruntime-extensions-src/build-shared/lib/libortextensions.so"),
        help="Host path to ORT Extensions shared library",
    )
    parser.add_argument(
        "--ortx-library-container-path",
        type=str,
        default="/opt/ortx/libortextensions.so",
        help="Container path for ORT Extensions library",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/parity_fused"),
        help="Directory for per-backbone parity reports",
    )
    return parser.parse_args()


def wait_ready(url: str, timeout_seconds: int) -> bool:
    start = time.time()
    while time.time() - start <= timeout_seconds:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def infer_triton(http_port: int, model_name: str, text: str) -> np.ndarray:
    url = f"http://localhost:{http_port}/v2/models/{model_name}/versions/1/infer"
    payload = {
        "inputs": [
            {
                "name": "text",
                "shape": [1],
                "datatype": "BYTES",
                "data": [text],
            }
        ],
        "outputs": [{"name": "logits"}],
    }
    response = requests.post(url, json=payload, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(f"Triton infer failed for {model_name}: {response.status_code} {response.text}")

    body = response.json()
    outputs = body.get("outputs", [])
    if not outputs:
        raise RuntimeError(f"Triton infer response missing outputs for {model_name}")

    output = outputs[0]
    shape = output.get("shape", [])
    data = output.get("data", [])
    if len(shape) != 2:
        raise RuntimeError(f"Unexpected Triton output shape for {model_name}: {shape}")

    logits = np.array(data, dtype=np.float32).reshape(shape)
    return logits


def load_config(model_dir: Path) -> Dict:
    config_path = model_dir / "chirpe_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as file:
        return json.load(file)


def build_model_inputs(model, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    accepted_names = set(inspect.signature(model.forward).parameters)
    filtered = {name: tensor for name, tensor in encoded_batch.items() if name in accepted_names}
    if "input_ids" not in filtered:
        raise ValueError("Filtered model inputs do not include input_ids")
    return filtered


def get_vote(predictions: np.ndarray) -> int:
    return int(np.bincount(predictions).argmax())


def run_backbone(
    backbone: str,
    model_name: str,
    args: argparse.Namespace,
    transcripts: List[Dict],
) -> Dict:
    checkpoint_dir = args.checkpoints_root / backbone / "best_model"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found for {backbone}: {checkpoint_dir}")

    config = load_config(checkpoint_dir)
    max_length = int(config.get("model", {}).get("max_length", 128))
    seg_threshold = float(config.get("preprocessing", {}).get("segmentation_threshold", 0.8))
    max_segments = int(config.get("preprocessing", {}).get("max_segments_per_transcript", 3))

    preprocessor = TranscriptPreprocessor(
        segmentation_threshold=seg_threshold,
        use_llm_summarizer=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            attn_implementation="eager",
        )
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    allclose = True
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    compared_values = 0
    segment_total = 0
    segment_match_count = 0
    transcript_total = 0
    transcript_match_count = 0
    skipped_no_segments = 0
    mismatch_examples = []

    for index, item in enumerate(transcripts):
        participant_id = item.get("participant_id", f"unknown_{index}")
        processed = preprocessor.process_transcript(item.get("transcript", []), participant_id)
        summaries = [segment["summary"] for segment in processed.get("segments", [])[:max_segments]]

        if not summaries:
            skipped_no_segments += 1
            continue

        encoded = tokenizer(
            summaries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        model_inputs = build_model_inputs(model, encoded)

        with torch.no_grad():
            torch_logits = model(**model_inputs).logits.cpu().numpy()

        triton_logits_list = []
        for text in summaries:
            triton_logits = infer_triton(args.http_port, model_name, text)
            triton_logits_list.append(triton_logits[0])
        triton_logits = np.stack(triton_logits_list, axis=0)

        abs_diff = np.abs(torch_logits - triton_logits)
        max_abs_diff = max(max_abs_diff, float(abs_diff.max()))
        sum_abs_diff += float(abs_diff.sum())
        compared_values += int(abs_diff.size)
        allclose = allclose and bool(np.allclose(torch_logits, triton_logits, atol=args.atol, rtol=args.rtol))

        torch_preds = np.argmax(torch_logits, axis=-1)
        triton_preds = np.argmax(triton_logits, axis=-1)
        segment_total += len(torch_preds)
        segment_match_count += int((torch_preds == triton_preds).sum())

        torch_vote = get_vote(torch_preds)
        triton_vote = get_vote(triton_preds)
        transcript_total += 1
        transcript_match_count += int(torch_vote == triton_vote)

        if torch_vote != triton_vote and len(mismatch_examples) < 10:
            mismatch_examples.append(
                {
                    "participant_id": participant_id,
                    "torch_segment_preds": torch_preds.tolist(),
                    "triton_segment_preds": triton_preds.tolist(),
                    "torch_vote": torch_vote,
                    "triton_vote": triton_vote,
                }
            )

    mean_abs_diff = (sum_abs_diff / compared_values) if compared_values else 0.0
    segment_agreement = (segment_match_count / segment_total) if segment_total else 0.0
    transcript_agreement = (
        transcript_match_count / transcript_total if transcript_total else 0.0
    )

    passed = (
        allclose
        and segment_agreement >= args.target_label_agreement
        and transcript_agreement >= args.target_transcript_agreement
    )

    report = {
        "status": "passed" if passed else "failed",
        "backbone": backbone,
        "model_name": model_name,
        "checkpoint_dir": str(checkpoint_dir),
        "data_file": str(args.data_file),
        "atol": args.atol,
        "rtol": args.rtol,
        "allclose": allclose,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "segment_total": segment_total,
        "segment_match_count": segment_match_count,
        "segment_agreement": segment_agreement,
        "target_segment_agreement": args.target_label_agreement,
        "transcript_total": transcript_total,
        "transcript_match_count": transcript_match_count,
        "transcript_agreement": transcript_agreement,
        "target_transcript_agreement": args.target_transcript_agreement,
        "skipped_no_segments": skipped_no_segments,
        "evaluated_transcripts": transcript_total,
        "mismatch_examples": mismatch_examples,
        "max_segments_per_transcript": max_segments,
        "segmentation_threshold": seg_threshold,
        "max_length": max_length,
    }

    return report


def main() -> None:
    args = parse_args()

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    if not backbones:
        raise SystemExit("No backbones provided")

    if not args.model_repo.exists():
        raise SystemExit(f"Model repository not found: {args.model_repo}")
    if not args.data_file.exists():
        raise SystemExit(f"Data file not found: {args.data_file}")

    with open(args.data_file, "r") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        transcripts = [payload]
    elif isinstance(payload, list):
        transcripts = payload
    else:
        raise SystemExit("Data file must contain a JSON object or list")

    if args.max_transcripts > 0:
        transcripts = transcripts[: args.max_transcripts]

    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise SystemExit("Docker CLI not found in PATH")

    ortx_library_host_path = args.ortx_library_host_path
    if not ortx_library_host_path.exists():
        raise SystemExit(f"ORT Extensions library path not found: {ortx_library_host_path}")

    run_cmd = [
        docker_bin,
        "run",
        "-d",
        "--net=host",
        "-v",
        f"{args.model_repo.resolve()}:/models",
        "-v",
        f"{ortx_library_host_path.resolve()}:{args.ortx_library_container_path}:ro",
        args.triton_image,
        "tritonserver",
        "--model-repository=/models",
    ]
    container_id = subprocess.check_output(run_cmd, text=True).strip()

    summary = {
        "status": "failed",
        "container_id": container_id,
        "model_repo": str(args.model_repo.resolve()),
        "checkpoints_root": str(args.checkpoints_root.resolve()),
        "data_file": str(args.data_file.resolve()),
        "num_transcripts_input": len(transcripts),
        "reports": [],
    }

    try:
        server_ready = wait_ready(
            f"http://localhost:{args.http_port}/v2/health/ready",
            args.timeout_seconds,
        )
        summary["server_ready"] = server_ready
        if not server_ready:
            raise RuntimeError("Triton server did not become ready")

        for backbone in backbones:
            model_name = BACKBONE_TO_MODEL.get(backbone)
            if not model_name:
                raise ValueError(f"Unsupported backbone: {backbone}")

            model_ready = wait_ready(
                f"http://localhost:{args.http_port}/v2/models/{model_name}/versions/1/ready",
                args.timeout_seconds,
            )
            if not model_ready:
                raise RuntimeError(f"Model not ready in Triton: {model_name}")

            report = run_backbone(backbone, model_name, args, transcripts)
            report_path = args.report_root / f"{backbone}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
            report["report_file"] = str(report_path)
            summary["reports"].append(report)

        summary["status"] = (
            "passed" if all(report.get("status") == "passed" for report in summary["reports"]) else "failed"
        )
    finally:
        try:
            logs = subprocess.check_output([docker_bin, "logs", container_id], text=True)
            summary["container_logs_tail"] = "\n".join(logs.splitlines()[-100:])
        except Exception:
            summary["container_logs_tail"] = ""
        subprocess.run([docker_bin, "stop", container_id], check=False, stdout=subprocess.DEVNULL)
        subprocess.run([docker_bin, "rm", "-f", container_id], check=False, stdout=subprocess.DEVNULL)

    summary_path = args.report_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
