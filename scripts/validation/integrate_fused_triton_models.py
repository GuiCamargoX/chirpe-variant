#!/usr/bin/env python3
"""Generate Triton configs and validate fused string-input models."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import onnxruntime as ort
import requests
from onnxruntime_extensions import get_library_path


ORT_TO_TRITON_DTYPE = {
    "tensor(bool)": "TYPE_BOOL",
    "tensor(double)": "TYPE_FP64",
    "tensor(float)": "TYPE_FP32",
    "tensor(float16)": "TYPE_FP16",
    "tensor(int16)": "TYPE_INT16",
    "tensor(int32)": "TYPE_INT32",
    "tensor(int64)": "TYPE_INT64",
    "tensor(int8)": "TYPE_INT8",
    "tensor(string)": "TYPE_STRING",
    "tensor(uint16)": "TYPE_UINT16",
    "tensor(uint32)": "TYPE_UINT32",
    "tensor(uint64)": "TYPE_UINT64",
    "tensor(uint8)": "TYPE_UINT8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate fused string-input ONNX models with Triton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path("outputs/string_onnx_fused"),
        help="Triton model repository root containing fused models",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        default="chirpe_bert_string,chirpe_clinicalbert_string,chirpe_mentalbert_string",
        help="Comma-separated fused model names",
    )
    parser.add_argument("--version", type=int, default=1, help="Model version directory")
    parser.add_argument(
        "--triton-image",
        type=str,
        default="nvcr.io/nvidia/tritonserver:24.10-py3",
        help="Triton docker image",
    )
    parser.add_argument("--http-port", type=int, default=8000, help="Triton HTTP port")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Readiness timeout")
    parser.add_argument(
        "--sample-text",
        type=str,
        default="The participant reports suspiciousness and mild confusion.",
        help="Sample text used for infer endpoint verification",
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
        help="Container path used in Triton model_operations op_library_filename",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("reports/triton_fused_integration/summary.json"),
        help="Output report JSON file",
    )
    return parser.parse_args()


def to_triton_dtype(ort_type: str) -> str:
    if ort_type not in ORT_TO_TRITON_DTYPE:
        raise ValueError(f"Unsupported ORT tensor type: {ort_type}")
    return ORT_TO_TRITON_DTYPE[ort_type]


def triton_dims(shape: Sequence) -> List[int]:
    dims: List[int] = []
    for dim in list(shape):
        if isinstance(dim, int):
            dims.append(dim if dim >= 0 else -1)
        else:
            dims.append(-1)
    return dims


def format_dims(dims: Sequence[int]) -> str:
    return ", ".join(str(value) for value in dims)


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


def infer_endpoint(http_port: int, model_name: str, sample_text: str) -> Dict:
    url = f"http://localhost:{http_port}/v2/models/{model_name}/versions/1/infer"
    payload = {
        "inputs": [
            {
                "name": "text",
                "shape": [1],
                "datatype": "BYTES",
                "data": [sample_text],
            }
        ],
        "outputs": [{"name": "logits"}],
    }
    response = requests.post(url, json=payload, timeout=15)
    if response.status_code != 200:
        return {
            "ok": False,
            "status_code": response.status_code,
            "error": response.text,
        }

    body = response.json()
    outputs = body.get("outputs", [])
    if not outputs:
        return {"ok": False, "status_code": 200, "error": "Missing outputs in infer response"}

    first_output = outputs[0]
    shape = first_output.get("shape", [])
    data = first_output.get("data", [])
    valid = bool(shape and len(shape) == 2 and shape[-1] == 2 and len(data) >= 2)

    return {
        "ok": valid,
        "status_code": 200,
        "output_name": first_output.get("name"),
        "output_shape": shape,
        "output_datatype": first_output.get("datatype"),
        "num_values": len(data),
    }


def generate_config_for_model(
    model_repo: Path,
    model_name: str,
    version: int,
    ortx_library_host_path: Path,
    ortx_library_container_path: str,
) -> Dict:
    model_root = model_repo / model_name
    version_dir = model_root / str(version)
    model_path = version_dir / "model.onnx"
    config_path = model_root / "config.pbtxt"

    if not model_path.exists():
        raise FileNotFoundError(f"Fused model not found: {model_path}")

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(str(ortx_library_host_path))
    session = ort.InferenceSession(str(model_path), session_options, providers=["CPUExecutionProvider"])

    onnx_inputs = session.get_inputs()
    onnx_outputs = session.get_outputs()

    if len(onnx_inputs) != 1:
        raise ValueError(f"Expected one input for fused model {model_name}, got {len(onnx_inputs)}")

    model_input = onnx_inputs[0]
    if to_triton_dtype(model_input.type) != "TYPE_STRING":
        raise ValueError(f"Expected TYPE_STRING input for {model_name}, got {model_input.type}")

    output_lines = []
    for index, model_output in enumerate(onnx_outputs):
        dtype = to_triton_dtype(model_output.type)
        dims = format_dims(triton_dims(model_output.shape))
        suffix = "," if index < len(onnx_outputs) - 1 else ""
        output_lines.append(
            f'  {{ name: "{model_output.name}" data_type: {dtype} dims: [ {dims} ] }}{suffix}'
        )

    config_text = "\n".join(
        [
            f'name: "{model_name}"',
            'backend: "onnxruntime"',
            "max_batch_size: 0",
            "",
            "input [",
            f'  {{ name: "{model_input.name}" data_type: TYPE_STRING dims: [ -1 ] }}',
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

    config_path.write_text(config_text)

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "input": {
            "name": model_input.name,
            "type": model_input.type,
            "shape": list(model_input.shape),
        },
        "outputs": [
            {
                "name": item.name,
                "type": item.type,
                "shape": list(item.shape),
            }
            for item in onnx_outputs
        ],
    }


def main() -> None:
    args = parse_args()

    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    if not model_names:
        raise SystemExit("No model names provided")

    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise SystemExit("Docker CLI not found in PATH")

    ortx_library_host_path = args.ortx_library_host_path
    if not ortx_library_host_path.exists():
        fallback = Path(get_library_path())
        if fallback.exists():
            ortx_library_host_path = fallback
        else:
            raise SystemExit(f"ORT Extensions library not found: {args.ortx_library_host_path}")

    if not args.model_repo.exists():
        raise SystemExit(f"Model repository not found: {args.model_repo}")

    report: Dict = {
        "status": "failed",
        "model_repo": str(args.model_repo.resolve()),
        "model_names": model_names,
        "version": args.version,
        "triton_image": args.triton_image,
        "http_port": args.http_port,
        "ortx_library_host_path": str(ortx_library_host_path.resolve()),
        "ortx_library_container_path": args.ortx_library_container_path,
    }

    generated_models = []
    for model_name in model_names:
        generated_models.append(
            generate_config_for_model(
                model_repo=args.model_repo,
                model_name=model_name,
                version=args.version,
                ortx_library_host_path=ortx_library_host_path,
                ortx_library_container_path=args.ortx_library_container_path,
            )
        )
    report["generated_models"] = generated_models

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
    report["container_id"] = container_id

    try:
        server_ready = wait_ready(
            f"http://localhost:{args.http_port}/v2/health/ready",
            args.timeout_seconds,
        )
        report["server_ready"] = server_ready

        model_status = []
        all_models_ready = True
        all_infer_ok = True
        for model_name in model_names:
            ready_url = f"http://localhost:{args.http_port}/v2/models/{model_name}/versions/{args.version}/ready"
            model_ready = wait_ready(ready_url, args.timeout_seconds)
            infer_result = infer_endpoint(args.http_port, model_name, args.sample_text)
            model_status.append(
                {
                    "model_name": model_name,
                    "ready": model_ready,
                    "infer": infer_result,
                }
            )
            all_models_ready = all_models_ready and model_ready
            all_infer_ok = all_infer_ok and bool(infer_result.get("ok"))

        report["model_status"] = model_status
        report["all_models_ready"] = all_models_ready
        report["all_models_infer_ok"] = all_infer_ok
        report["status"] = "passed" if server_ready and all_models_ready and all_infer_ok else "failed"
    finally:
        try:
            logs = subprocess.check_output([docker_bin, "logs", container_id], text=True)
            report["container_logs_tail"] = "\n".join(logs.splitlines()[-100:])
        except Exception:
            report["container_logs_tail"] = ""
        subprocess.run([docker_bin, "stop", container_id], check=False, stdout=subprocess.DEVNULL)
        subprocess.run([docker_bin, "rm", "-f", container_id], check=False, stdout=subprocess.DEVNULL)

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
