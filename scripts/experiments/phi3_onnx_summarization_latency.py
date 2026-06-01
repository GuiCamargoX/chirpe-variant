#!/usr/bin/env python3
"""Smoke test local Phi-3 ONNX summarization latency on CPU.

This script is intentionally a local experiment. It downloads a CPU-friendly
ONNX Runtime GenAI Phi-3 variant into outputs/ and asks it to summarize one
example segment. It does not modify the CHiRPE model graph.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


MODEL_ID = "microsoft/Phi-3-mini-4k-instruct-onnx"
MODEL_SUBDIR = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
DEFAULT_OUTPUT_DIR = Path("outputs/local_onnx_llm/phi3-mini-4k-instruct-onnx")
DEFAULT_TEXT = """
The participant reports that ordinary events sometimes feel personally meaningful
and connected to them. They describe occasional suspiciousness and worries that
others may be watching them, but they also say these experiences are intermittent
and not always distressing. They deny current intent to harm themselves or others,
and they remain able to attend school and complete daily activities with some
difficulty concentrating.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Phi-3 ONNX Runtime GenAI summarization latency smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--download", action="store_true", help="Download the model if needed")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=42)
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Text to summarize")
    return parser.parse_args()


def require_genai():
    try:
        import onnxruntime_genai as og  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: onnxruntime-genai. Install with: "
            "conda run --no-capture-output -n chirp python -m pip install onnxruntime-genai"
        ) from exc
    return og


def download_model(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=[f"{MODEL_SUBDIR}/*"],
        local_dir=output_dir,
    )
    elapsed = time.perf_counter() - started
    print(f"Download/check time: {elapsed:.2f}s")
    return output_dir / MODEL_SUBDIR


def find_model_dir(output_dir: Path) -> Path:
    model_dir = output_dir / MODEL_SUBDIR
    if not model_dir.exists():
        raise SystemExit(
            "Model directory is missing. Re-run with --download to fetch the CPU int4 ONNX model."
        )
    return model_dir


def build_prompt(text: str) -> str:
    instruction = (
        "Summarize the clinical interview segment in exactly 2 concise sentences. "
        "Focus on symptoms, functional impact, and risk-relevant details. "
        "Return only the summary text and do not repeat yourself."
    )
    return f"<|user|>\n{instruction}\n\nSegment:\n{text}<|end|>\n<|assistant|>"


def decode_tokens(stream, tokens: Iterable[int]) -> str:
    return "".join(stream.decode(token) for token in tokens)


def main() -> None:
    args = parse_args()
    og = require_genai()

    model_dir = download_model(args.output_dir) if args.download else find_model_dir(args.output_dir)

    prompt = build_prompt(args.text)
    print("Model: Phi-3 mini 4k instruct ONNX CPU int4")
    print("Model directory: outputs/local_onnx_llm/.../cpu-int4-rtn-block-32-acc-level-4")
    print(f"Input characters: {len(args.text)}")
    print(f"Requested max new tokens: {args.max_new_tokens}")

    load_started = time.perf_counter()
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()
    load_elapsed = time.perf_counter() - load_started

    tokenize_started = time.perf_counter()
    input_tokens = tokenizer.encode(prompt)
    tokenize_elapsed = time.perf_counter() - tokenize_started

    prompt_token_count = len(input_tokens)
    max_length = prompt_token_count + args.max_new_tokens

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, batch_size=1)
    generator = og.Generator(model, params)

    append_started = time.perf_counter()
    generator.append_tokens(input_tokens)
    append_elapsed = time.perf_counter() - append_started

    generated_tokens = []
    first_token_latency = None
    generation_started = time.perf_counter()

    while not generator.is_done():
        generator.generate_next_token()
        next_token = generator.get_next_tokens()[0]
        if first_token_latency is None:
            first_token_latency = time.perf_counter() - generation_started
        generated_tokens.append(int(next_token))

    generation_elapsed = time.perf_counter() - generation_started
    end_to_end_generation_elapsed = append_elapsed + generation_elapsed
    summary = decode_tokens(stream, generated_tokens).strip()
    tokens_per_second = len(generated_tokens) / generation_elapsed if generation_elapsed else 0.0

    print("\nSummary:")
    print(summary)
    print("\nLatency:")
    print(f"Model load time: {load_elapsed:.2f}s")
    print(f"Prompt tokenization time: {tokenize_elapsed:.3f}s")
    print(f"Prompt append/prefill time: {append_elapsed:.3f}s")
    print(f"Prompt tokens: {prompt_token_count}")
    print(f"Generated tokens: {len(generated_tokens)}")
    print(f"First-token latency: {(first_token_latency or 0.0):.3f}s")
    print(f"Total generation time: {generation_elapsed:.2f}s")
    print(f"End-to-end generation time: {end_to_end_generation_elapsed:.2f}s")
    print(f"Generation throughput: {tokens_per_second:.2f} tokens/s")


if __name__ == "__main__":
    main()
