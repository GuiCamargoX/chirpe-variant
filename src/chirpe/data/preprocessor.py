"""Main preprocessing pipeline for CHiRPE."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from chirpe.data.segmentation import Segment, SymptomSegmenter
from chirpe.data.summarizer import (
    Phi3OnnxSummarizer,
    SegmentSummarizer,
    SimpleSummarizer,
)

logger = logging.getLogger(__name__)


class TranscriptPreprocessor:
    """Pipeline that segments transcripts and summarizes each mapped segment.

    This class is the bridge between raw interview transcripts and classifier-
    ready text snippets.
    """

    def __init__(
        self,
        segmentation_threshold: float = 0.80,
        use_llm_summarizer: bool = False,
        llm_model_name: Optional[str] = None,
        summarizer_backend: str = "hf",
        phi3_model_dir: Optional[str] = None,
        phi3_download_root: Optional[str] = None,
        phi3_max_new_tokens: int = 64,
        phi3_download: bool = False,
    ):
        """Initialize the preprocessor.

        Args:
            segmentation_threshold: Threshold for fuzzy matching
            use_llm_summarizer: Whether to use an LLM for summarization. When
                False, the deterministic ``SimpleSummarizer`` is used.
            llm_model_name: Model name for the HuggingFace ``SegmentSummarizer``
                (used when ``summarizer_backend == "hf"``).
            summarizer_backend: Which LLM backend to use when
                ``use_llm_summarizer`` is True: ``"hf"`` for a local
                HuggingFace causal LM, or ``"phi3_onnx"`` for the local Phi-3
                ONNX Runtime GenAI summarizer.
            phi3_model_dir: Explicit ONNX GenAI model directory for the Phi-3
                backend (overrides ``phi3_download_root``).
            phi3_download_root: Root directory the Phi-3 ONNX model is loaded
                from / downloaded into.
            phi3_max_new_tokens: Max new tokens per segment for the Phi-3 backend.
            phi3_download: Whether to download the Phi-3 ONNX model if missing.
        """
        self.segmenter = SymptomSegmenter(threshold=segmentation_threshold)

        if use_llm_summarizer and summarizer_backend == "phi3_onnx":
            self.summarizer = Phi3OnnxSummarizer(
                model_dir=phi3_model_dir,
                download_root=phi3_download_root,
                max_new_tokens=phi3_max_new_tokens,
                download=phi3_download,
            )
        elif use_llm_summarizer and llm_model_name:
            self.summarizer = SegmentSummarizer(
                model_name=llm_model_name,
            )
        else:
            self.summarizer = SimpleSummarizer()

        logger.info(
            "Initialized TranscriptPreprocessor with %s summarizer",
            type(self.summarizer).__name__,
        )

    def process_transcript(
        self, transcript: List[Dict[str, str]], participant_id: str = ""
    ) -> Dict:
        """Process a single transcript through the pipeline.

        Args:
            transcript: List of utterance dicts
            participant_id: Participant identifier

        Returns:
            Dictionary with keys:
            - `participant_id`
            - `segments`: list of per-domain dictionaries containing text,
              summary, and positional metadata
            - `num_segments`
            - `domains_covered`
        """
        # Step 1: Segment by symptom domain
        segments = self.segmenter.segment_transcript(transcript)

        # Step 2: Summarize each mapped segment.
        segment_data = []
        for segment in segments:
            # Unmapped buckets are useful for diagnostics but excluded from
            # downstream training/prediction payloads.
            if segment.domain == "unmapped":
                continue

            summary = self.summarizer.summarize_segment(segment.get_text())
            segment_data.append({
                "domain": segment.domain,
                "text": segment.get_text(),
                "summary": summary,
                "start_idx": segment.start_idx,
                "end_idx": segment.end_idx,
                "utterance_count": len(segment.utterances),
            })

        return {
            "participant_id": participant_id,
            "segments": segment_data,
            "num_segments": len(segment_data),
            "domains_covered": list(set(s["domain"] for s in segment_data)),
        }

    def process_dataset(
        self,
        data: List[Dict],
        output_dir: Optional[Path] = None,
    ) -> List[Dict]:
        """Process a full dataset.

        Args:
            data: List of transcript data
            output_dir: Optional directory where `processed_data.json` is
                written.

        Returns:
            List of processed data
        """
        processed_data = []

        for item in data:
            participant_id = item.get("participant_id", "unknown")
            transcript = item.get("transcript", [])
            label = item.get("label", "Unknown")

            logger.debug(f"Processing transcript for {participant_id}")

            processed = self.process_transcript(transcript, participant_id)
            processed["label"] = label
            processed["original"] = item

            processed_data.append(processed)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "processed_data.json", "w") as f:
                json.dump(processed_data, f, indent=2)

            logger.info(f"Saved processed data to {output_dir}")

        return processed_data

    def get_summary_texts(self, processed_data: Dict) -> List[str]:
        """Extract summary texts from processed data.

        Args:
            processed_data: Output from process_transcript

        Returns:
            List of summary texts
        """
        return [seg["summary"] for seg in processed_data.get("segments", [])]

    def get_concatenated_summary(self, processed_data: Dict) -> str:
        """Concatenate all segment summaries into one classifier input string.

        Args:
            processed_data: Output from process_transcript

        Returns:
            Space-delimited summary text.
        """
        summaries = self.get_summary_texts(processed_data)
        return " ".join(summaries)
