"""Tests for segmentation module."""

import pytest

from chirpe.data.segmentation import SymptomSegmenter, Segment


class TestSymptomSegmenter:
    """Test SymptomSegmenter class."""

    def test_initialization(self):
        """Test segmenter initialization."""
        segmenter = SymptomSegmenter(threshold=0.80)
        assert segmenter.threshold == 80

    def test_match_utterance_to_domain(self):
        """Test matching utterances to domains."""
        segmenter = SymptomSegmenter(threshold=0.80)

        # Test P1 matching
        domain = segmenter._match_utterance_to_domain(
            "Have you ever felt the radio was communicating with you?"
        )
        assert domain == "P1_Unusual_Thoughts"

        # Test P2 matching
        domain = segmenter._match_utterance_to_domain(
            "Do you feel suspicious of people around you?"
        )
        assert domain == "P2_Suspiciousness"

        # Test no match
        domain = segmenter._match_utterance_to_domain(
            "What did you have for breakfast today?"
        )
        assert domain is None

    def test_segment_transcript(self):
        """Test transcript segmentation."""
        segmenter = SymptomSegmenter(threshold=0.80)

        transcript = [
            {"speaker": "interviewer", "text": "Have you heard voices others couldn't hear?"},
            {"speaker": "interviewee", "text": "Yes, sometimes I hear things."},
            {"speaker": "interviewer", "text": "Do you feel suspicious of people?"},
            {"speaker": "interviewee", "text": "Sometimes I do."},
        ]

        segments = segmenter.segment_transcript(transcript)

        assert len(segments) > 0
        assert all(isinstance(s, Segment) for s in segments)

    def test_get_segment_by_domain(self):
        """Test retrieving segment by domain."""
        segmenter = SymptomSegmenter(threshold=0.80)

        segments = [
            Segment(domain="P1_Unusual_Thoughts", utterances=[], start_idx=0, end_idx=1),
            Segment(domain="P2_Suspiciousness", utterances=[], start_idx=2, end_idx=3),
        ]

        p1_segment = segmenter.get_segment_by_domain(segments, "P1_Unusual_Thoughts")
        assert p1_segment is not None
        assert p1_segment.domain == "P1_Unusual_Thoughts"

        missing = segmenter.get_segment_by_domain(segments, "P3_Unusual_Somatic")
        assert missing is None

    def test_get_all_domain_texts(self):
        """Test getting all domain texts."""
        segmenter = SymptomSegmenter(threshold=0.80)

        segments = [
            Segment(
                domain="P1_Unusual_Thoughts",
                utterances=[{"speaker": "interviewee", "text": "I hear voices"}],
                start_idx=0,
                end_idx=0,
            ),
            Segment(
                domain="P1_Unusual_Thoughts",
                utterances=[{"speaker": "interviewee", "text": "I see things"}],
                start_idx=1,
                end_idx=1,
            ),
        ]

        domain_texts = segmenter.get_all_domain_texts(segments)
        assert "P1_Unusual_Thoughts" in domain_texts
        assert "I hear voices I see things" == domain_texts["P1_Unusual_Thoughts"]
