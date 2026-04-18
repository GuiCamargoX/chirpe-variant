"""LLM-based summarization of interview segments."""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default prompts based on the paper
FIRST_PASS_PROMPT = """You are an expert clinical interviewer. Summarise the following interview segment in a single third person paragraph, covering what was asked and the detailed response.

Interview segment: {segment}

Draft summary:"""

SECOND_PASS_PROMPT = """Here is a transcript segment and an initial draft summary. Improve the summary by adding any important information from the segment that was missed. Keep third person narration, one coherent paragraph, and no bullet points.

Interview segment: {segment}

Draft summary: {draft}

Improved summary:"""


class SegmentSummarizer:
    """Summarize interview segments using either local or API-backed LLMs."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        use_api: bool = False,
        api_key: Optional[str] = None,
        device: str = "auto",
    ):
        """Initialize the summarizer.

        Args:
            model_name: HuggingFace model name or "openai" for GPT
            use_api: Whether to use API-based model
            api_key: API key for OpenAI/Anthropic
            device: Device to load model on
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        self.device = device
        self.model = None
        self.tokenizer = None

        if not use_api:
            self._load_local_model()

    def _load_local_model(self):
        """Load a local causal LM and text-generation pipeline."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            logger.info(f"Loading model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Configure model loading based on model type
            load_kwargs = {
                "torch_dtype": "auto",
                "device_map": self.device,
            }

            # Add Gemma-specific configuration
            if "gemma" in self.model_name.lower():
                # Gemma models may need specific attention implementation
                load_kwargs["attn_implementation"] = "eager"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # Set padding token if not present (needed for some models like Gemma)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate_local(self, prompt: str, max_length: int = 512) -> str:
        """Generate text with the local HF pipeline.

        Args:
            prompt: Prompt text.
            max_length: Maximum number of new tokens to generate.

        Returns:
            Generated text (prompt excluded).
        """
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

    def _generate_api(self, prompt: str, max_length: int = 512) -> str:
        """Generate text via API-backed models.

        Note:
            This path currently uses the legacy `openai.ChatCompletion.create`
            interface.
        """
        if "openai" in self.model_name.lower():
            import openai

            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert clinical interviewer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_length,
                temperature=0.7,
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported API model: {self.model_name}")

    def summarize_segment(self, segment_text: str) -> str:
        """Summarize a segment using two-pass approach.

        Args:
            segment_text: The text to summarize

        Returns:
            One-paragraph third-person summary.
        """
        if self.use_api:
            # Simplified single-pass for API
            prompt = FIRST_PASS_PROMPT.format(segment=segment_text)
            summary = self._generate_api(prompt)
            return summary

        # First pass: produce a faithful draft.
        first_prompt = FIRST_PASS_PROMPT.format(segment=segment_text)
        draft = self._generate_local(first_prompt)

        # Second pass: repair omissions and improve coherence.
        second_prompt = SECOND_PASS_PROMPT.format(
            segment=segment_text,
            draft=draft,
        )
        final_summary = self._generate_local(second_prompt)

        return final_summary

    def summarize_segments(self, segments: List["Segment"]) -> List[str]:
        """Summarize multiple segments.

        Args:
            segments: List of Segment objects

        Returns:
            List of summaries
        """
        summaries = []
        for segment in segments:
            text = segment.get_text()
            summary = self.summarize_segment(text)
            summaries.append(summary)
        return summaries


class SimpleSummarizer:
    """Deterministic fallback summarizer for testing and quick demos."""

    def __init__(self):
        """Initialize the simple summarizer."""
        pass

    def summarize_segment(self, segment_text: str) -> str:
        """Create a simple third-person summary.

        Args:
            segment_text: The text to summarize

        Returns:
            Simple third-person summary
        """
        # Heuristic removal of obvious interviewer questions.
        lines = segment_text.split("\n")
        interviewee_lines = []

        for line in lines:
            # Simple heuristic: look for interviewee indicators
            if not line.lower().startswith(("have you", "do you", "are you", "can you")):
                interviewee_lines.append(line)

        content = " ".join(interviewee_lines)

        # Basic first-person to third-person substitutions.
        content = content.replace("I ", "The interviewee ")
        content = content.replace("I'm ", "The interviewee is ")
        content = content.replace("I've ", "The interviewee has ")
        content = content.replace("my ", "their ")
        content = content.replace("My ", "Their ")
        content = content.replace("me ", "them ")
        content = content.replace("Me ", "Them ")

        # Keep fallback summaries short and predictable.
        if len(content.split()) > 100:
            content = " ".join(content.split()[:100]) + "..."

        return content

    def summarize_segments(self, segments: List["Segment"]) -> List[str]:
        """Summarize multiple segments.

        Args:
            segments: List of Segment objects

        Returns:
            List of summaries
        """
        return [self.summarize_segment(s.get_text()) for s in segments]
