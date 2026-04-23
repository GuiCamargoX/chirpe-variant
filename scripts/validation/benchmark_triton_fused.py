#!/usr/bin/env python3
"""Benchmark fused Triton string-input model serving metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests


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


@dataclass
class MemorySample:
    peak_bytes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused Triton string-input model metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-repo", type=Path, required=True, help="Triton model repository root")
    parser.add_argument("--model-name", type=str, required=True, help="Model name in Triton repo")
    parser.add_argument("--report-file", type=Path, required=True, help="Output JSON report path")
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Optional classifier-only baseline report for direct comparison",
    )
    parser.add_argument(
        "--triton-image",
        type=str,
        default="nvcr.io/nvidia/tritonserver:24.10-py3",
        help="Triton docker image",
    )
    parser.add_argument("--http-port", type=int, default=8000, help="Triton HTTP port")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8",
        help="Comma-separated concurrency levels (reported as batch sizes)",
    )
    parser.add_argument("--warmup-requests", type=int, default=10, help="Warmup requests per batch size")
    parser.add_argument("--requests", type=int, default=60, help="Measured requests per batch size")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="Infer request timeout seconds")
    parser.add_argument("--ready-timeout", type=int, default=120, help="Triton readiness timeout seconds")
    parser.add_argument(
        "--memory-poll-interval",
        type=float,
        default=0.5,
        help="Seconds between docker stats memory polls",
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
        help="Container mount path for ORT Extensions shared library",
    )
    return parser.parse_args()


def parse_bytes(value: str) -> int:
    match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?i?B)\s*$", value)
    if not match:
        raise ValueError(f"Unsupported memory format: {value}")

    magnitude = float(match.group(1))
    unit = match.group(2)
    factors = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
        "PiB": 1024**5,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "PB": 1000**5,
    }
    return int(magnitude * factors[unit])


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.2f} {units[index]}"


def wait_ready(url: str, timeout_seconds: int) -> None:
    start = time.time()
    while time.time() - start <= timeout_seconds:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for endpoint: {url}")


def memory_sampler(container_id: str, poll_interval: float, stop_event: threading.Event, sample: MemorySample) -> None:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        return

    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                [docker_bin, "stats", "--no-stream", "--format", "{{.MemUsage}}", container_id],
                text=True,
            ).strip()
            if output:
                current = parse_bytes(output.split("/")[0].strip())
                sample.peak_bytes = max(sample.peak_bytes, current)
        except Exception:
            pass
        stop_event.wait(poll_interval)


def build_payload(texts: List[str]) -> Dict:
    return {
        "inputs": [
            {
                "name": "text",
                "shape": [len(texts)],
                "datatype": "BYTES",
                "data": texts,
            }
        ],
        "outputs": [{"name": "logits"}],
    }


def infer_once(
    http_port: int,
    model_name: str,
    text: str,
    request_timeout: float,
) -> float:
    infer_url = f"http://localhost:{http_port}/v2/models/{model_name}/versions/1/infer"
    payload = build_payload([text])

    start = time.perf_counter()
    response = requests.post(infer_url, json=payload, timeout=request_timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if response.status_code != 200:
        raise RuntimeError(f"Infer request failed ({response.status_code}): {response.text}")

    outputs = response.json().get("outputs", [])
    if not outputs:
        raise RuntimeError("Infer response missing outputs")
    shape = outputs[0].get("shape", [])
    if len(shape) != 2 or int(shape[0]) != 1 or int(shape[1]) != 2:
        raise RuntimeError(f"Unexpected output shape for single-text request: {shape}")

    return elapsed_ms


def benchmark_batch_size(
    http_port: int,
    model_name: str,
    batch_size: int,
    warmup_requests: int,
    measured_requests: int,
    request_timeout: float,
) -> Dict[str, float]:
    latencies_ms: List[float] = []
    measured_total_samples = 0
    total_warmup_requests = warmup_requests * batch_size
    total_measured_requests = measured_requests * batch_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        for request_index in range(total_warmup_requests):
            text = DEFAULT_TEXTS[request_index % len(DEFAULT_TEXTS)]
            infer_once(
                http_port=http_port,
                model_name=model_name,
                text=text,
                request_timeout=request_timeout,
            )

        measured_start = time.perf_counter()
        wave = 0
        while measured_total_samples < total_measured_requests:
            futures = []
            for offset in range(batch_size):
                index = wave * batch_size + offset
                text = DEFAULT_TEXTS[index % len(DEFAULT_TEXTS)]
                futures.append(
                    executor.submit(
                        infer_once,
                        http_port,
                        model_name,
                        text,
                        request_timeout,
                    )
                )

            for future in futures:
                latencies_ms.append(float(future.result()))
            measured_total_samples += batch_size
            wave += 1

        measured_end = time.perf_counter()

    latency_array = np.array(latencies_ms, dtype=np.float64)
    elapsed = max(measured_end - measured_start, 1e-9)
    return {
        "batch_size": batch_size,
        "num_requests": len(latencies_ms),
        "mode": "concurrency",
        "p50_latency_ms": float(np.percentile(latency_array, 50)),
        "p95_latency_ms": float(np.percentile(latency_array, 95)),
        "mean_latency_ms": float(np.mean(latency_array)),
        "throughput_samples_per_sec": float(measured_total_samples / elapsed),
    }


def compare_with_baseline(fused_metrics: List[Dict], baseline_report: Path) -> Dict:
    if not baseline_report.exists():
        raise FileNotFoundError(f"Baseline report not found: {baseline_report}")

    with open(baseline_report, "r") as file:
        baseline_payload = json.load(file)

    baseline_metrics = {item["batch_size"]: item for item in baseline_payload.get("metrics", [])}
    comparisons = []
    for fused in fused_metrics:
        batch_size = fused["batch_size"]
        baseline = baseline_metrics.get(batch_size)
        if not baseline:
            continue

        comparisons.append(
            {
                "batch_size": batch_size,
                "baseline_p50_latency_ms": baseline["p50_latency_ms"],
                "fused_p50_latency_ms": fused["p50_latency_ms"],
                "delta_p50_latency_ms": fused["p50_latency_ms"] - baseline["p50_latency_ms"],
                "baseline_throughput_samples_per_sec": baseline["throughput_samples_per_sec"],
                "fused_throughput_samples_per_sec": fused["throughput_samples_per_sec"],
                "delta_throughput_samples_per_sec": (
                    fused["throughput_samples_per_sec"] - baseline["throughput_samples_per_sec"]
                ),
            }
        )

    return {
        "baseline_report": str(baseline_report),
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()

    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise SystemExit("Docker CLI not found in PATH")
    if not args.model_repo.exists():
        raise SystemExit(f"Model repository not found: {args.model_repo}")
    if not args.ortx_library_host_path.exists():
        raise SystemExit(f"ORT Extensions library path not found: {args.ortx_library_host_path}")

    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    if not batch_sizes:
        raise SystemExit("No batch sizes provided")

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

    stop_event = threading.Event()
    sample = MemorySample()
    sampler_started = False
    sampler_thread = threading.Thread(
        target=memory_sampler,
        args=(container_id, args.memory_poll_interval, stop_event, sample),
        daemon=True,
    )

    report: Dict = {
        "status": "failed",
        "model_name": args.model_name,
        "model_repo": str(args.model_repo.resolve()),
        "triton_image": args.triton_image,
        "http_port": args.http_port,
        "batch_sizes": batch_sizes,
        "mode": "concurrency",
        "mode_note": "Fused tokenizer op currently handles one string per infer request; non-1 request tensors are not supported.",
        "warmup_requests": args.warmup_requests,
        "measured_requests": args.requests,
        "container_id": container_id,
    }

    try:
        wait_ready(f"http://localhost:{args.http_port}/v2/health/ready", args.ready_timeout)
        wait_ready(
            f"http://localhost:{args.http_port}/v2/models/{args.model_name}/versions/1/ready",
            args.ready_timeout,
        )

        sampler_thread.start()
        sampler_started = True

        metrics = []
        for batch_size in batch_sizes:
            metrics.append(
                benchmark_batch_size(
                    http_port=args.http_port,
                    model_name=args.model_name,
                    batch_size=batch_size,
                    warmup_requests=args.warmup_requests,
                    measured_requests=args.requests,
                    request_timeout=args.request_timeout,
                )
            )

        report.update(
            {
                "status": "passed",
                "metrics": metrics,
                "memory": {
                    "peak_container_memory_bytes": sample.peak_bytes,
                    "peak_container_memory_human": format_bytes(sample.peak_bytes),
                },
            }
        )

        if args.baseline_report:
            report["baseline_comparison"] = compare_with_baseline(metrics, args.baseline_report)

    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    finally:
        stop_event.set()
        if sampler_started:
            sampler_thread.join(timeout=2)
        try:
            logs = subprocess.check_output([docker_bin, "logs", container_id], text=True)
            report["container_logs_tail"] = "\n".join(logs.splitlines()[-80:])
        except Exception:
            report["container_logs_tail"] = ""
        subprocess.run([docker_bin, "stop", container_id], check=False, stdout=subprocess.DEVNULL)
        subprocess.run([docker_bin, "rm", "-f", container_id], check=False, stdout=subprocess.DEVNULL)

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if report.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
