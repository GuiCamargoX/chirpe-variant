#!/usr/bin/env python3
"""Run robustness checks for fused Triton string-input models."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness checks for fused Triton string-input models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path("outputs/string_onnx_fused"),
        help="Triton model repository with fused models",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        default="chirpe_bert_string,chirpe_clinicalbert_string,chirpe_mentalbert_string",
        help="Comma-separated model names",
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
        "--concurrency-workers",
        type=int,
        default=8,
        help="Concurrent workers for stress checks",
    )
    parser.add_argument(
        "--requests-per-worker",
        type=int,
        default=20,
        help="Requests per worker in concurrency check",
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
        help="Container path for ORT Extensions shared library",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("reports/robustness_fused/summary.json"),
        help="Output report path",
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


def infer(http_port: int, model_name: str, payload: Dict, timeout: float = 20.0) -> requests.Response:
    url = f"http://localhost:{http_port}/v2/models/{model_name}/versions/1/infer"
    return requests.post(url, json=payload, timeout=timeout)


def valid_payload(text: str) -> Dict:
    return {
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


def check_valid_cases(http_port: int, model_name: str) -> Dict:
    cases = {
        "empty": "",
        "whitespace": "   ",
        "punctuation": "!!! ??? ... ;;;",
        "unicode": "Paciente relata deja vu ocasional e confusao leve. 你好.",
        "long": " ".join(["symptom"] * 3000),
    }
    results = []
    all_ok = True

    for name, text in cases.items():
        response = infer(http_port, model_name, valid_payload(text))
        case_ok = response.status_code == 200
        shape = []
        if case_ok:
            outputs = response.json().get("outputs", [])
            if outputs:
                shape = outputs[0].get("shape", [])
            case_ok = bool(len(shape) == 2 and shape[0] == 1 and shape[1] == 2)
        else:
            all_ok = False

        results.append(
            {
                "case": name,
                "status_code": response.status_code,
                "ok": case_ok,
                "shape": shape,
                "error": "" if response.status_code == 200 else response.text,
            }
        )
        all_ok = all_ok and case_ok

    return {"ok": all_ok, "results": results}


def check_malformed_cases(http_port: int, model_name: str) -> Dict:
    malformed_payloads = [
        {"name": "missing_inputs", "payload": {}},
        {
            "name": "wrong_input_name",
            "payload": {
                "inputs": [{"name": "bad", "shape": [1], "datatype": "BYTES", "data": ["x"]}],
                "outputs": [{"name": "logits"}],
            },
        },
        {
            "name": "wrong_datatype",
            "payload": {
                "inputs": [{"name": "text", "shape": [1], "datatype": "INT64", "data": [1]}],
                "outputs": [{"name": "logits"}],
            },
        },
        {
            "name": "shape_mismatch",
            "payload": {
                "inputs": [{"name": "text", "shape": [2], "datatype": "BYTES", "data": ["one"]}],
                "outputs": [{"name": "logits"}],
            },
        },
    ]

    results = []
    all_ok = True
    for item in malformed_payloads:
        response = infer(http_port, model_name, item["payload"])
        expected_failure = response.status_code >= 400
        results.append(
            {
                "case": item["name"],
                "status_code": response.status_code,
                "ok": expected_failure,
                "error": "" if response.status_code == 200 else response.text,
            }
        )
        all_ok = all_ok and expected_failure

    return {"ok": all_ok, "results": results}


def check_concurrency(http_port: int, model_name: str, workers: int, requests_per_worker: int) -> Dict:
    failures: List[str] = []

    def worker(worker_idx: int) -> int:
        success = 0
        for request_idx in range(requests_per_worker):
            text = f"worker={worker_idx} request={request_idx} occasional suspiciousness"
            response = infer(http_port, model_name, valid_payload(text), timeout=30.0)
            if response.status_code != 200:
                failures.append(f"{worker_idx}:{request_idx}:{response.status_code}")
                continue
            outputs = response.json().get("outputs", [])
            if not outputs:
                failures.append(f"{worker_idx}:{request_idx}:missing_outputs")
                continue
            shape = outputs[0].get("shape", [])
            if len(shape) != 2 or shape[0] != 1 or shape[1] != 2:
                failures.append(f"{worker_idx}:{request_idx}:shape={shape}")
                continue
            success += 1
        return success

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        success_counts = list(executor.map(worker, range(workers)))

    total = workers * requests_per_worker
    success = sum(success_counts)
    return {
        "ok": success == total and not failures,
        "workers": workers,
        "requests_per_worker": requests_per_worker,
        "total_requests": total,
        "successful_requests": success,
        "failed_requests": total - success,
        "failure_examples": failures[:20],
    }


def main() -> None:
    args = parse_args()
    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    if not model_names:
        raise SystemExit("No model names provided")

    if not args.model_repo.exists():
        raise SystemExit(f"Model repository not found: {args.model_repo}")
    if not args.ortx_library_host_path.exists():
        raise SystemExit(f"ORT Extensions library path not found: {args.ortx_library_host_path}")

    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise SystemExit("Docker CLI not found in PATH")

    run_cmd = [
        docker_bin,
        "run",
        "-d",
        "--net=host",
        "-v",
        f"{args.model_repo.resolve()}:/models",
        "-v",
        f"{args.ortx_library_host_path.resolve()}:{args.ortx_library_container_path}:ro",
        args.triton_image,
        "tritonserver",
        "--model-repository=/models",
    ]
    container_id = subprocess.check_output(run_cmd, text=True).strip()

    report: Dict = {
        "status": "failed",
        "container_id": container_id,
        "model_repo": str(args.model_repo.resolve()),
        "model_names": model_names,
        "checks": [],
    }

    try:
        server_ready = wait_ready(
            f"http://localhost:{args.http_port}/v2/health/ready",
            args.timeout_seconds,
        )
        report["server_ready"] = server_ready
        if not server_ready:
            raise RuntimeError("Triton server did not become ready")

        for model_name in model_names:
            model_ready = wait_ready(
                f"http://localhost:{args.http_port}/v2/models/{model_name}/versions/1/ready",
                args.timeout_seconds,
            )
            if not model_ready:
                raise RuntimeError(f"Model not ready: {model_name}")

            valid_result = check_valid_cases(args.http_port, model_name)
            malformed_result = check_malformed_cases(args.http_port, model_name)
            concurrency_result = check_concurrency(
                args.http_port,
                model_name,
                workers=args.concurrency_workers,
                requests_per_worker=args.requests_per_worker,
            )

            model_ok = valid_result["ok"] and malformed_result["ok"] and concurrency_result["ok"]
            report["checks"].append(
                {
                    "model_name": model_name,
                    "status": "passed" if model_ok else "failed",
                    "valid_cases": valid_result,
                    "malformed_cases": malformed_result,
                    "concurrency": concurrency_result,
                }
            )

        report["status"] = "passed" if all(item["status"] == "passed" for item in report["checks"]) else "failed"
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
