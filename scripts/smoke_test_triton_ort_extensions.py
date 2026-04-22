#!/usr/bin/env python3
"""Create and optionally run a Triton custom-op smoke test for ORT Extensions."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import requests
from onnx import save
from onnxruntime_extensions import gen_processing_models, get_library_path
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tokenizer ONNX model and verify Triton can load ORT Extensions custom ops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path("outputs/triton_ortx_smoketest"),
        help="Triton model repository path to create",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ortx_tokenizer_smoketest",
        help="Triton model name",
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=1,
        help="Triton model version directory",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face tokenizer used to generate a tokenizer ONNX model",
    )
    parser.add_argument(
        "--triton-image",
        type=str,
        default="nvcr.io/nvidia/tritonserver:24.10-py3",
        help="Triton docker image",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8000,
        help="Triton HTTP port",
    )
    parser.add_argument(
        "--run-docker",
        action="store_true",
        help="Run Triton in Docker and perform readiness checks",
    )
    parser.add_argument(
        "--ortx-library-host-path",
        type=Path,
        help="Host path to libortextensions shared library to mount into Triton",
    )
    parser.add_argument(
        "--ortx-library-container-path",
        type=str,
        default="/opt/ortx/libortextensions.so",
        help="Container path used in Triton config for mounted ORT Extensions library",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="Max wait for Triton to become ready",
    )
    return parser.parse_args()


def build_model_repository(
    model_repo: Path,
    model_name: str,
    model_version: int,
    hf_tokenizer: str,
    extension_library_container_path: str,
) -> tuple[Path, Path]:
    model_root = model_repo / model_name
    version_dir = model_root / str(model_version)
    version_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    pre_model, _ = gen_processing_models(tokenizer, pre_kwargs={})
    save(pre_model, str(version_dir / "model.onnx"))

    config_pbtxt = f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 0

input [
  {{ name: "text" data_type: TYPE_STRING dims: [ -1 ] }}
]

output [
  {{ name: "input_ids" data_type: TYPE_INT64 dims: [ -1 ] }},
  {{ name: "token_type_ids" data_type: TYPE_INT64 dims: [ -1 ] }},
  {{ name: "attention_mask" data_type: TYPE_INT64 dims: [ -1 ] }},
  {{ name: "offset_mapping" data_type: TYPE_INT64 dims: [ -1, 2 ] }}
]

model_operations: {{
  op_library_filename: ["{extension_library_container_path}"]
}}

instance_group [
  {{ kind: KIND_CPU }}
]
'''

    (model_root / "config.pbtxt").write_text(config_pbtxt)

    details = {
        "model_repo": str(model_repo.resolve()),
        "model_name": model_name,
        "model_version": model_version,
        "hf_tokenizer": hf_tokenizer,
        "onnx_path": str((version_dir / "model.onnx").resolve()),
        "config_path": str((model_root / "config.pbtxt").resolve()),
        "requires_custom_op_library": extension_library_container_path,
    }
    (model_root / "smoketest_metadata.json").write_text(json.dumps(details, indent=2))

    return model_root, version_dir


def check_endpoint(url: str, timeout_seconds: int) -> bool:
    start = time.time()
    while time.time() - start <= timeout_seconds:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(1)
            continue
        time.sleep(1)
    return False


def run_docker_smoke_test(
    model_repo: Path,
    extension_library_host_path: Path,
    extension_library_container_path: str,
    triton_image: str,
    model_name: str,
    model_version: int,
    http_port: int,
    timeout_seconds: int,
) -> dict:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise RuntimeError("Docker is not available in PATH.")

    run_cmd = [
        docker_bin,
        "run",
        "-d",
        "--net=host",
        "-v",
        f"{model_repo.resolve()}:/models",
        "-v",
        f"{extension_library_host_path.resolve()}:{extension_library_container_path}:ro",
        triton_image,
        "tritonserver",
        "--model-repository=/models",
    ]

    container_id = subprocess.check_output(run_cmd, text=True).strip()

    ready_url = f"http://localhost:{http_port}/v2/health/ready"
    model_ready_url = (
        f"http://localhost:{http_port}/v2/models/{model_name}/versions/{model_version}/ready"
    )

    result: dict = {
        "container_id": container_id,
        "ready_url": ready_url,
        "model_ready_url": model_ready_url,
        "server_ready": False,
        "model_ready": False,
    }

    try:
        result["server_ready"] = check_endpoint(ready_url, timeout_seconds)
        result["model_ready"] = check_endpoint(model_ready_url, timeout_seconds)

        try:
            logs = subprocess.check_output([docker_bin, "logs", container_id], text=True)
            result["container_logs_tail"] = "\n".join(logs.splitlines()[-60:])
        except subprocess.CalledProcessError as exc:
            result["container_logs_tail"] = ""
            result["log_error"] = str(exc)
    finally:
        subprocess.run([docker_bin, "stop", container_id], check=False, stdout=subprocess.DEVNULL)
        subprocess.run([docker_bin, "rm", "-f", container_id], check=False, stdout=subprocess.DEVNULL)

    return result


def main() -> None:
    args = parse_args()

    extension_library_host_path = (
        args.ortx_library_host_path if args.ortx_library_host_path else Path(get_library_path())
    )
    extension_library_container_path = args.ortx_library_container_path

    if not extension_library_host_path.exists():
        raise SystemExit(
            f"ORT Extensions library not found at: {extension_library_host_path}"
        )

    model_root, version_dir = build_model_repository(
        model_repo=args.model_repo,
        model_name=args.model_name,
        model_version=args.model_version,
        hf_tokenizer=args.hf_tokenizer,
        extension_library_container_path=extension_library_container_path,
    )

    report = {
        "status": "prepared",
        "model_repo": str(args.model_repo.resolve()),
        "model_root": str(model_root.resolve()),
        "version_dir": str(version_dir.resolve()),
        "extension_library_host_path": str(extension_library_host_path.resolve()),
        "extension_library_container_path": extension_library_container_path,
    }

    if args.run_docker:
        docker_result = run_docker_smoke_test(
            model_repo=args.model_repo,
            extension_library_host_path=extension_library_host_path,
            extension_library_container_path=extension_library_container_path,
            triton_image=args.triton_image,
            model_name=args.model_name,
            model_version=args.model_version,
            http_port=args.http_port,
            timeout_seconds=args.timeout_seconds,
        )
        report["docker_smoke_test"] = docker_result
        if docker_result["server_ready"] and docker_result["model_ready"]:
            report["status"] = "passed"
        else:
            report["status"] = "failed"

    print(json.dumps(report, indent=2))

    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
