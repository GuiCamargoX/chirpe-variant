"""Probe the Phi-3 ONNX summarizer for hallucination.

For each segment we generate WITHOUT the stop-marker cleanup and with a larger
token budget, so we can see what the model produces past the requested 2-sentence
summary. Then we apply the production cleanup (``_clean_output``) and report:

  - the raw generation,
  - which stop marker fired (i.e. what hallucinated continuation was trimmed),
  - the cleaned output that the pipeline actually uses.

Run:
    conda run --no-capture-output -n chirp python scripts/experiments/phi3_hallucination_probe.py
"""

from __future__ import annotations

from pathlib import Path

from chirpe.data.segmentation import SymptomSegmenter
from chirpe.data.summarizer import Phi3OnnxSummarizer

MODEL_DIR = Path(
    "outputs/local_onnx_llm/phi3-mini-4k-instruct-onnx/"
    "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
)

DEMO_TRANSCRIPT = [
    {
        "speaker": "interviewer",
        "text": "Tell me about whether events have had special meaning for you.",
    },
    {
        "speaker": "interviewee",
        "text": "Sometimes I feel like television messages are personally directed at me.",
    },
    {
        "speaker": "interviewer",
        "text": "Tell me whether people are talking about you or watching you.",
    },
    {
        "speaker": "interviewee",
        "text": "At times I worry strangers are watching me when I am outside, but it comes and goes.",
    },
    {"speaker": "interviewer", "text": "Tell me about any trouble concentrating or focusing."},
    {
        "speaker": "interviewee",
        "text": "I have mild trouble focusing at school, but I do not get lost or confused about where I am.",
    },
]


def generate_raw(summarizer: Phi3OnnxSummarizer, segment_text: str, max_new_tokens: int) -> str:
    """Generate text with the same params as production but no stop-marker cleanup."""
    og = summarizer._og
    prompt = summarizer._build_prompt(segment_text)
    if summarizer.tokenizer_backend == "hf":
        input_tokens = summarizer.hf_tokenizer.encode(prompt)
    else:
        input_tokens = summarizer.og_tokenizer.encode(prompt)

    params = og.GeneratorParams(summarizer.model)
    params.set_search_options(
        max_length=len(input_tokens) + max_new_tokens,
        batch_size=1,
        do_sample=False,
        num_beams=1,
    )
    generator = og.Generator(summarizer.model, params)
    generator.append_tokens(input_tokens)

    generated_ids = []
    while not generator.is_done():
        generator.generate_next_token()
        generated_ids.append(int(generator.get_next_tokens()[0]))

    if summarizer.tokenizer_backend == "hf":
        return summarizer.hf_tokenizer.decode(generated_ids, skip_special_tokens=False)
    stream = summarizer.og_tokenizer.create_stream()
    return "".join(stream.decode(t) for t in generated_ids)


def which_marker(raw: str):
    """Return (marker, index) of the earliest stop marker present in raw, else None."""
    hits = [(m, raw.find(m)) for m in Phi3OnnxSummarizer._STOP_MARKERS if raw.find(m) != -1]
    return min(hits, key=lambda x: x[1]) if hits else None


def main() -> None:
    summarizer = Phi3OnnxSummarizer(
        model_dir=str(MODEL_DIR), max_new_tokens=64, tokenizer_backend="hf"
    )
    segments = [
        s
        for s in SymptomSegmenter(threshold=0.80).segment_transcript(DEMO_TRANSCRIPT)
        if s.domain != "unmapped"
    ]

    # Generate with a generous budget so the model has room to "keep going"
    # past the 2-sentence summary if it is inclined to hallucinate.
    raw_budget = 200

    for i, seg in enumerate(segments, 1):
        text = seg.get_text()
        raw = generate_raw(summarizer, text, max_new_tokens=raw_budget)
        cleaned = Phi3OnnxSummarizer._clean_output(raw)
        marker = which_marker(raw)

        print("=" * 78)
        print(f"SEGMENT {i}  domain={seg.domain}")
        print(f"  input: {text[:110]!r}")
        print(f"\n  RAW generation ({raw_budget} token budget):")
        print("    " + raw.replace("\n", "\\n"))
        if marker:
            m, idx = marker
            tail = raw[idx:].replace("\n", "\\n")
            print(f"\n  >> STOP MARKER fired: {m!r} at char {idx}")
            print(f"     HALLUCINATED/trimmed tail: {tail[:200]!r}")
        else:
            print("\n  >> No stop marker fired (model stopped on its own / hit token budget).")
        print(f"\n  CLEANED (what the pipeline uses):")
        print(f"    {cleaned!r}")
        print()


if __name__ == "__main__":
    main()
