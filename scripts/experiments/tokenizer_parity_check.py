"""Token-ID parity check: og.Tokenizer vs HF AutoTokenizer for the Phi-3 prompt.

This is the *fair* test of the question Juon raised: do the two tokenizer
implementations produce identical token IDs for the same prompt? Both read the
same ``tokenizer.json`` shipped with the model, so they are *supposed* to agree.
If they agree on every input, the downstream is provably identical (same IDs ->
same generation -> same summary -> same classifier input -> same CHR-P decision),
so no classifier comparison is needed. If they diverge anywhere, this script
prints exactly where, and those cases are the only ones worth checking downstream.

We deliberately do NOT route this through the (untrained-on-this-data) classifier,
because that would launder a clean, deterministic question through a noisy
component.

Usage:
    conda run --no-capture-output -n chirp python scripts/experiments/tokenizer_parity_check.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import List

from chirpe.data.segmentation import SymptomSegmenter
from chirpe.data.summarizer import PHI3_INSTRUCTION, Phi3OnnxSummarizer

MODEL_DIR = Path(
    "outputs/local_onnx_llm/phi3-mini-4k-instruct-onnx/"
    "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
)

# Adversarial edge cases: the kinds of inputs where two tokenizer
# implementations are most likely to disagree (special markers, odd whitespace,
# unicode, repeated punctuation, code-like text).
EDGE_CASES = [
    "",
    " ",
    "\n\n\t  leading and trailing whitespace  \n",
    "Patient said: <|user|> then <|end|> appeared mid-text.",
    "Voices — they whisper… café, naïve, fiancé. 🙂 ¿Qué?",
    "Numbers 1234567890 and symbols !@#$%^&*()_+-=[]{}|;':\",./<>?",
    "Repeated punctuation!!!??? and    multiple     spaces.",
    "A very long run of words " + "symptom " * 200,
    "Mixed\r\nline\rendings\nhere",
    "ZeroWidth​Space and NBSP here",
]


def _build_prompt(text: str) -> str:
    """Use the exact same prompt construction the summarizer uses."""
    return f"<|user|>\n{PHI3_INSTRUCTION}\n\nSegment:\n{text}<|end|>\n<|assistant|>"


def collect_segment_texts() -> List[str]:
    """Segment every synthetic transcript and collect the raw segment texts."""
    segmenter = SymptomSegmenter(threshold=0.80)
    texts: List[str] = []
    for path in sorted(glob.glob("data/**/*.json", recursive=True)):
        try:
            records = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(records, list):
            continue
        for rec in records:
            transcript = rec.get("transcript") if isinstance(rec, dict) else None
            if not transcript:
                continue
            for seg in segmenter.segment_transcript(transcript):
                if seg.domain != "unmapped":
                    texts.append(seg.get_text())
    return texts


def main() -> None:
    import onnxruntime_genai as og
    from transformers import AutoTokenizer

    print(f"Loading og model + tokenizers from {MODEL_DIR}")
    model = og.Model(str(MODEL_DIR))
    og_tok = og.Tokenizer(model)
    hf_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    seg_texts = collect_segment_texts()
    print(f"Collected {len(seg_texts)} real segment texts + {len(EDGE_CASES)} edge cases")

    inputs = [("segment", t) for t in seg_texts] + [("edge", t) for t in EDGE_CASES]
    prompts = [_build_prompt(t) for _, t in inputs]

    mismatches = []
    for (kind, raw), prompt in zip(inputs, prompts):
        og_ids = [int(x) for x in og_tok.encode(prompt)]
        hf_ids = list(hf_tok.encode(prompt))
        if og_ids != hf_ids:
            # first index where they differ
            first = next(
                (i for i, (a, b) in enumerate(zip(og_ids, hf_ids)) if a != b),
                min(len(og_ids), len(hf_ids)),
            )
            mismatches.append((kind, raw, og_ids, hf_ids, first))

    total = len(prompts)
    n_match = total - len(mismatches)
    print("\n" + "=" * 60)
    print(f"RESULT: {n_match}/{total} prompts produced identical token IDs")
    print(f"        ({len(mismatches)} mismatch(es))")
    print("=" * 60)

    if not mismatches:
        print(
            "\nog.Tokenizer and HF AutoTokenizer are EQUIVALENT on all inputs.\n"
            "=> Switching to HF is provably safe; downstream (summary + CHR-P)\n"
            "   is identical, so the classifier comparison is unnecessary."
        )
        return

    print("\nDivergences (these are the only cases worth a downstream check):")
    for kind, raw, og_ids, hf_ids, first in mismatches:
        lo, hi = max(0, first - 2), first + 3
        print(f"\n[{kind}] len(og)={len(og_ids)} len(hf)={len(hf_ids)} first diff @ {first}")
        print(f"  input (repr, first 120 chars): {raw[:120]!r}")
        print(f"  og[{lo}:{hi}] = {og_ids[lo:hi]}")
        print(f"  hf[{lo}:{hi}] = {hf_ids[lo:hi]}")


if __name__ == "__main__":
    main()
