"""Main preprocessing pipeline for CHiRPE."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from chirpe.data.segmentation import Segment, SymptomSegmenter
from chirpe.data.summarizer import SegmentSummarizer, SimpleSummarizer

logger = logging.getLogger(__name__)


class TranscriptPreprocessor:
    """Main preprocessing pipeline for PSYCHS transcripts."""

    def __init__(
        self,
        segmentation_threshold: float = 0.80,
        use_llm_summarizer: bool = False,
        llm_model_name: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
    ):
        """Initialize the preprocessor.

        Args:
            segmentation_threshold: Threshold for fuzzy matching
            use_llm_summarizer: Whether to use LLM for summarization
            llm_model_name: Model name for LLM summarizer
            use_api: Whether to use API-based LLM
            api_key: API key for LLM
        """
        self.segmenter = SymptomSegmenter(threshold=segmentation_threshold)

        if use_llm_summarizer and llm_model_name:
            self.summarizer = SegmentSummarizer(
                model_name=llm_model_name,
                use_api=use_api,
                api_key=api_key,
            )
        else:
            self.summarizer = SimpleSummarizer()

        logger.info(
            f"Initialized TranscriptPreprocessor with "
            f"{'LLM' if use_llm_summarizer else 'simple'} summarizer"
        )

    def process_transcript(
        self, transcript: List[Dict[str, str]], participant_id: str = ""
    ) -> Dict:
        """Process a single transcript through the pipeline.

        Args:
            transcript: List of utterance dicts
            participant_id: Participant identifier

        Returns:
            Processed data with segments and summaries
        """
        # Step 1: Segment by symptom domain
        segments = self.segmenter.segment_transcript(transcript)

        # Step 2: Summarize each segment
        segment_data = []
        for segment in segments:
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
            output_dir: Optional directory to save processed data

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
        """Get all summaries concatenated.

        Args:
            processed_data: Output from process_transcript

        Returns:
            Concatenated summary text
        """
        summaries = self.get_summary_texts(processed_data)
        return " ".join(summaries)
