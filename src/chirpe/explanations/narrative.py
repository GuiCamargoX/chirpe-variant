"""Narrative explanation generation using LLMs."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NARRATIVE_PROMPT = """You are an expert clinical interviewer. Rewrite the excerpt into ONE clinician-friendly paragraph (max 3 sentences) describing {symptom}.

Excerpt: {excerpt}

Narrative:"""

QUOTE_PROMPT = """Provide ONLY the interviewee quote (enclosed in double quotation marks) that clearly illustrates {symptom} and supports the narrative. Output the quote and nothing else.

Transcript: {transcript}

Quote:"""


class NarrativeGenerator:
    """Generate clinical narrative explanations."""

    def __init__(
        self,
        model_name: str = "qwen3-4b",
        use_api: bool = False,
        api_key: Optional[str] = None,
    ):
        """Initialize the narrative generator.

        Args:
            model_name: Model name (qwen3-4b, gpt-4, etc.)
            use_api: Whether to use API-based generation
            api_key: API key for LLM
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        self.model = None
        self.tokenizer = None

        if not use_api:
            self._load_model()

    def _load_model(self):
        """Load the local model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading narrative model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Narrative model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load narrative model: {e}")
            raise

    def _generate_local(self, prompt: str, max_length: int = 256) -> str:
        """Generate using local model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part after the prompt
        return generated[len(prompt) :].strip()

    def _generate_api(self, prompt: str) -> str:
        """Generate using API."""
        if "gpt" in self.model_name.lower():
            import openai

            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model="gpt-4" if "4" in self.model_name else "gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert clinical interviewer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unsupported API model: {self.model_name}")

    def generate_narrative(
        self,
        excerpt: str,
        symptom: str,
        include_quote: bool = True,
        full_transcript: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate narrative summary for a symptom.

        Args:
            excerpt: Text excerpt to summarize
            symptom: Symptom name
            include_quote: Whether to include a representative quote
            full_transcript: Full transcript for quote extraction

        Returns:
            Dictionary with narrative and quote
        """
        # Generate narrative
        prompt = NARRATIVE_PROMPT.format(symptom=symptom, excerpt=excerpt)

        if self.use_api:
            narrative = self._generate_api(prompt)
        else:
            narrative = self._generate_local(prompt)

        result = {"narrative": narrative, "symptom": symptom}

        # Generate quote if requested
        if include_quote and full_transcript:
            quote_prompt = QUOTE_PROMPT.format(
                symptom=symptom, transcript=full_transcript
            )
            if self.use_api:
                quote = self._generate_api(quote_prompt)
            else:
                quote = self._generate_local(quote_prompt)

            # Clean up quote
            quote = quote.strip().strip('"')
            result["quote"] = quote

        return result

    def generate_all_narratives(
        self,
        segments: List[Dict],
        top_k: int = 3,
    ) -> List[Dict[str, str]]:
        """Generate narratives for top-k segments.

        Args:
            segments: List of segment dicts with 'summary' and 'domain' keys
            top_k: Number of top segments to generate narratives for

        Returns:
            List of narrative dicts
        """
        narratives = []

        for segment in segments[:top_k]:
            narrative = self.generate_narrative(
                excerpt=segment["summary"],
                symptom=segment["domain"].replace("_", " "),
                include_quote=False,
            )
            narratives.append(narrative)

        return narratives


class SimpleNarrativeGenerator:
    """Simple rule-based narrative generator for testing."""

    def __init__(self):
        """Initialize simple generator."""
        pass

    def generate_narrative(
        self,
        excerpt: str,
        symptom: str,
        include_quote: bool = False,
        full_transcript: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate simple narrative.

        Args:
            excerpt: Text excerpt
            symptom: Symptom name
            include_quote: Ignored in simple version
            full_transcript: Ignored in simple version

        Returns:
            Simple narrative dict
        """
        # Simple template-based generation
        narrative = f"The interviewee reports experiences related to {symptom.replace('_', ' ')}. They describe {excerpt[:200]}..."

        result = {
            "narrative": narrative,
            "symptom": symptom,
        }

        if include_quote and full_transcript:
            # Extract first sentence as quote
            sentences = full_transcript.split(".")
            if sentences:
                result["quote"] = sentences[0].strip() + "."

        return result

    def generate_all_narratives(
        self,
        segments: List[Dict],
        top_k: int = 3,
    ) -> List[Dict[str, str]]:
        """Generate narratives for segments.

        Args:
            segments: List of segments
            top_k: Number to generate

        Returns:
            List of narratives
        """
        return [
            self.generate_narrative(
                excerpt=seg["summary"],
                symptom=seg["domain"],
            )
            for seg in segments[:top_k]
        ]
