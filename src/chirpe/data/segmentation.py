"""Symptom domain segmentation using fuzzy string matching."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

# PSYCHS Template Questions mapped to symptom domains
PSYCHS_DOMAINS = {
    "P1_Unusual_Thoughts": [
        "odd is going on",
        "something is wrong",
        "real or imaginary",
        "daydreamed",
        "fantasies",
        "experience of time",
        "seemed strange",
        "world have changed",
        "predict the future",
        "special meaning",
        "communicating directly",
        "superstitious",
        "controlling",
        "thoughts",
        "read your mind",
    ],
    "P2_Suspiciousness": [
        "talking about you",
        "laughing at you",
        "mistrustful",
        "suspicious",
        "pay close attention",
        "singled out",
        "watched",
        "giving you a hard time",
        "hurt you",
        "hostile",
    ],
    "P3_Unusual_Somatic": [
        "wrong with your body",
        "body shape",
        "body",
        "health",
    ],
    "P4_Disoriented": [
        "confused",
        "lose track",
        "concentrate",
        "focus",
        "attention",
    ],
    "P5_Focused_Thoughts": [
        "thoughts race",
        "ideas fast",
        "mind jumping",
        "distracted",
    ],
    "P6_Experiences": [
        "heard voices",
        "seeing things",
        "hallucinations",
        "perceptions",
    ],
    "P7_Perceiving": [
        "noises",
        "buzzing",
        "sounds",
        "sensations",
    ],
    "P8_Confusion": [
        "uncertain",
        "puzzled",
        "perplexed",
        "muddled",
    ],
    "P9_Daydreaming": [
        "fantasies",
        "imaginary",
        "lost in thought",
        "absorbed",
    ],
    "P10_Time_Experience": [
        "time changed",
        "faster",
        "slower",
        "time standing still",
    ],
    "P11_Depersonalisation": [
        "not actually exist",
        "outside yourself",
        "watching yourself",
        "not real",
    ],
    "P12_Derealisation": [
        "world not exist",
        "unreal",
        "dreamlike",
        "movie",
    ],
    "P13_Superstitious": [
        "omens",
        "signs",
        "magical thinking",
        "sixth sense",
    ],
    "P14_Control": [
        "force outside",
        "controlling",
        "interfering",
        "put into your head",
    ],
    "P15_Reading_Minds": [
        "read your mind",
        "know what you are thinking",
        "read other people's minds",
        "broadcast",
    ],
}


@dataclass
class Segment:
    """A segmented portion of a transcript."""

    domain: str
    utterances: List[Dict[str, str]]
    start_idx: int
    end_idx: int

    def get_text(self) -> str:
        """Get concatenated text from all utterances in this segment."""
        return " ".join([u["text"] for u in self.utterances])


class SymptomSegmenter:
    """Segment transcripts into PSYCHS domains using fuzzy keyword matching.

    The segmenter treats interviewer turns as boundary cues. When an
    interviewer utterance strongly matches a new domain, subsequent utterances
    are grouped into a new segment until another domain transition is detected.
    """

    def __init__(
        self,
        threshold: float = 0.80,
        domains: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize the segmenter.

        Args:
            threshold: Fuzzy matching threshold (0-1)
            domains: Custom domain mappings (uses PSYCHS_DOMAINS if None)
        """
        # FuzzyWuzzy scores are 0-100 integers while config uses 0-1 floats.
        self.threshold = int(threshold * 100)
        self.domains = domains or PSYCHS_DOMAINS
        logger.info(f"Initialized SymptomSegmenter with threshold {threshold}")

    def _match_utterance_to_domain(self, text: str) -> Optional[str]:
        """Match an utterance to a symptom domain.

        Args:
            text: The utterance text to match

        Returns:
            Best matching domain, or `None` when no keyword similarity exceeds
            the configured threshold.
        """
        text_lower = text.lower()
        best_match = None
        best_score = 0

        for domain, keywords in self.domains.items():
            for keyword in keywords:
                # Use partial ratio for matching substrings
                score = fuzz.partial_ratio(keyword.lower(), text_lower)
                if score > best_score and score >= self.threshold:
                    best_score = score
                    best_match = domain

        return best_match

    def segment_transcript(
        self, transcript: List[Dict[str, str]]
    ) -> List[Segment]:
        """Segment a transcript into symptom domains.

        Args:
            transcript: List of utterance dicts with 'speaker' and 'text' keys

        Returns:
            List of Segment objects
        """
        segments = []
        current_domain: Optional[str] = None
        current_utterances: List[Dict[str, str]] = []
        start_idx = 0

        for i, utterance in enumerate(transcript):
            if utterance.get("speaker") == "interviewer":
                # Interviewer prompts drive domain transitions.
                matched_domain = self._match_utterance_to_domain(utterance["text"])

                if matched_domain and matched_domain != current_domain:
                    # Save previous segment before switching domains.
                    if current_utterances:
                        segments.append(
                            Segment(
                                domain=current_domain or "unmapped",
                                utterances=current_utterances,
                                start_idx=start_idx,
                                end_idx=i - 1,
                            )
                        )
                    # Start collecting a new domain segment.
                    current_domain = matched_domain
                    current_utterances = [utterance]
                    start_idx = i
                else:
                    current_utterances.append(utterance)
            else:
                # Interviewee turns are attached to the current domain context.
                current_utterances.append(utterance)

        # Finalize trailing segment after loop completion.
        if current_utterances:
            segments.append(
                Segment(
                    domain=current_domain or "unmapped",
                    utterances=current_utterances,
                    start_idx=start_idx,
                    end_idx=len(transcript) - 1,
                )
            )

        logger.debug(f"Segmented transcript into {len(segments)} segments")
        return segments

    def get_segment_by_domain(
        self, segments: List[Segment], domain: str
    ) -> Optional[Segment]:
        """Get a specific segment by domain name.

        Args:
            segments: List of segments
            domain: Domain name to find

        Returns:
            Matching segment or None
        """
        for segment in segments:
            if segment.domain == domain:
                return segment
        return None

    def get_all_domain_texts(self, segments: List[Segment]) -> Dict[str, str]:
        """Get concatenated text for each domain.

        Args:
            segments: List of segments

        Returns:
            Dict mapping domain to text
        """
        domain_texts: Dict[str, List[str]] = {}
        for segment in segments:
            if segment.domain not in domain_texts:
                domain_texts[segment.domain] = []
            domain_texts[segment.domain].append(segment.get_text())

        return {k: " ".join(v) for k, v in domain_texts.items()}
