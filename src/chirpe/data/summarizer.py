"""LLM-based summarization of interview segments."""

import logging
import re
from pathlib import Path
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
    """Summarize interview segments using a local HuggingFace causal LM."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "auto",
    ):
        """Initialize the summarizer.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

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

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

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

    def summarize_segment(self, segment_text: str) -> str:
        """Summarize a segment using two-pass approach.

        Args:
            segment_text: The text to summarize

        Returns:
            One-paragraph third-person summary.
        """
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


# Instruction used to steer the SmolLM2 summarizer toward short, clinically
# focused, third-person summaries suitable as classifier input. Mirrors
# ``PHI3_INSTRUCTION`` so the two lightweight summarizers are interchangeable.
SMOLLM2_INSTRUCTION = (
    "Summarize the following clinical interview segment in exactly 2 concise, "
    "third-person sentences. Focus on symptoms, functional impact, and "
    "risk-relevant details. Return only the summary text and do not repeat yourself."
)


class SmolLM2Summarizer:
    """Lightweight summarizer backed by a SmolLM2 instruct model via transformers.

    SmolLM2 is a small (135M / 360M / 1.7B) instruction-tuned causal LM. The
    360M default keeps it a genuinely *light* alternative to the 7B
    ``SegmentSummarizer`` and the Phi-3 ONNX summarizer, while still running the
    same single-paragraph, third-person summarization contract used elsewhere in
    the pipeline.

    Decoding is greedy and deterministic (``do_sample=False``) so repeated runs
    over the same segment produce identical summaries.
    """

    DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        max_new_tokens: int = 64,
    ):
        """Initialize the SmolLM2 summarizer.

        Args:
            model_name: HuggingFace model id of a SmolLM2 instruct checkpoint.
            device: Device map passed to ``from_pretrained`` (e.g. ``"auto"``,
                ``"cpu"``, ``"cuda"``).
            max_new_tokens: Maximum number of new tokens generated per segment.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load the SmolLM2 tokenizer and causal LM."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading SmolLM2 summarizer {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

            logger.info("SmolLM2 summarizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SmolLM2 model: {e}")
            raise

    @staticmethod
    def _build_messages(segment_text: str) -> List[dict]:
        """Build the chat-format messages for one segment."""
        return [
            {"role": "system", "content": SMOLLM2_INSTRUCTION},
            {"role": "user", "content": f"Segment:\n{segment_text}"},
        ]

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip whitespace and any leaked chat-template markers from generation."""
        for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def summarize_segment(self, segment_text: str) -> str:
        """Summarize a single segment with SmolLM2.

        Args:
            segment_text: The text to summarize

        Returns:
            Short third-person clinical summary.
        """
        messages = self._build_messages(segment_text)
        # return_dict=True yields an explicit attention_mask; without it,
        # generate() warns and may misbehave because pad_token == eos_token.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        input_ids = inputs["input_ids"]

        output_ids = self.model.generate(
            input_ids,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Decode only the newly generated tokens (exclude the prompt).
        generated = output_ids[0][input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return self._clean_output(text)

    def summarize_segments(self, segments: List["Segment"]) -> List[str]:
        """Summarize multiple segments.

        Args:
            segments: List of Segment objects

        Returns:
            List of summaries
        """
        return [self.summarize_segment(s.get_text()) for s in segments]


# Instruction used to steer the Phi-3 ONNX summarizer toward short, clinically
# focused, third-person summaries suitable as classifier input.
PHI3_INSTRUCTION = (
    "Summarize the clinical interview segment in exactly 2 concise sentences. "
    "Focus on symptoms, functional impact, and risk-relevant details. "
    "Return only the summary text and do not repeat yourself."
)


class Phi3OnnxSummarizer:
    """Summarize interview segments with a local Phi-3 ONNX Runtime GenAI model.

    Wraps the CPU int4 ONNX build of ``microsoft/Phi-3-mini-4k-instruct-onnx``
    via ``onnxruntime-genai``. The model is loaded once and reused across all
    segments; each segment is summarized with a fresh generator.
    """

    MODEL_ID = "microsoft/Phi-3-mini-4k-instruct-onnx"
    MODEL_SUBDIR = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    DEFAULT_DOWNLOAD_ROOT = Path("outputs/local_onnx_llm/phi3-mini-4k-instruct-onnx")

    def __init__(
        self,
        model_dir: Optional[str] = None,
        download_root: Optional[str] = None,
        max_new_tokens: int = 64,
        download: bool = False,
        tokenizer_backend: str = "og",
    ):
        """Initialize the Phi-3 ONNX summarizer.

        Args:
            model_dir: Path to a directory containing the ONNX GenAI model
                (the folder with ``genai_config.json``). If given, it is used
                directly and ``download_root``/``download`` are ignored.
            download_root: Root directory the model is downloaded into / loaded
                from. Defaults to ``outputs/local_onnx_llm/phi3-mini-4k-instruct-onnx``.
                The model is expected under ``<download_root>/<MODEL_SUBDIR>``.
            max_new_tokens: Maximum number of new tokens to generate per segment.
            download: Whether to fetch the model from the Hugging Face Hub if it
                is not already present.
            tokenizer_backend: Which tokenizer turns the prompt into token IDs.
                ``"og"`` (default) uses ``onnxruntime_genai.Tokenizer``;
                ``"hf"`` uses ``transformers.AutoTokenizer`` loaded from the same
                model directory. The og.Generator runs the model in both cases —
                only the tokenizer changes. Both read the same ``tokenizer.json``
                and produce identical IDs on natural-language input (verified by
                ``scripts/experiments/tokenizer_parity_check.py``); the ``"hf"``
                backend exists because ``onnxruntime_genai`` has no Triton support,
                whereas ``AutoTokenizer`` is a standard, deployable dependency.
        """
        if tokenizer_backend not in ("og", "hf"):
            raise ValueError(f"tokenizer_backend must be 'og' or 'hf', got {tokenizer_backend!r}")
        self.max_new_tokens = max_new_tokens
        self.tokenizer_backend = tokenizer_backend
        self._og = self._require_genai()

        resolved_dir = self._resolve_model_dir(model_dir, download_root, download)
        logger.info(
            f"Loading Phi-3 ONNX summarizer from {resolved_dir} "
            f"(tokenizer_backend={tokenizer_backend})"
        )
        self.model = self._og.Model(str(resolved_dir))
        # The og.Generator always runs the model; only the tokenizer differs.
        self.og_tokenizer = self._og.Tokenizer(self.model)
        if tokenizer_backend == "hf":
            from transformers import AutoTokenizer

            self.hf_tokenizer = AutoTokenizer.from_pretrained(str(resolved_dir))
        else:
            self.hf_tokenizer = None
        logger.info("Phi-3 ONNX summarizer loaded successfully")

    @staticmethod
    def _require_genai():
        """Import onnxruntime-genai with a helpful error if it is missing."""
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: onnxruntime-genai. Install with: "
                "python -m pip install onnxruntime-genai"
            ) from exc
        return og

    @classmethod
    def _resolve_model_dir(
        cls,
        model_dir: Optional[str],
        download_root: Optional[str],
        download: bool,
    ) -> Path:
        """Resolve (and optionally download) the ONNX GenAI model directory."""
        if model_dir is not None:
            path = Path(model_dir)
            if not path.exists():
                raise FileNotFoundError(f"Phi-3 ONNX model directory not found: {path}")
            return path

        root = Path(download_root) if download_root else cls.DEFAULT_DOWNLOAD_ROOT
        target = root / cls.MODEL_SUBDIR
        if not target.exists():
            if not download:
                raise FileNotFoundError(
                    f"Phi-3 ONNX model not found at {target}. Pass download=True "
                    "to fetch it from the Hugging Face Hub, or set model_dir."
                )
            from huggingface_hub import snapshot_download

            root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {cls.MODEL_ID} ({cls.MODEL_SUBDIR}) to {root}")
            snapshot_download(
                repo_id=cls.MODEL_ID,
                allow_patterns=[f"{cls.MODEL_SUBDIR}/*"],
                local_dir=root,
            )
        return target

    # Markers that signal the model has stopped summarizing and started
    # hallucinating a continuation (chat-template turns, a new transcript, or a
    # fresh paragraph/section after the requested short summary).
    _STOP_MARKERS = (
        "<|end|>",
        "<|user|>",
        "<|assistant|>",
        "\nInterviewer:",
        "\nInterviewee:",
        "\n\n",
    )

    @staticmethod
    def _build_prompt(text: str) -> str:
        """Wrap segment text in the Phi-3 chat template with the summary instruction."""
        return f"<|user|>\n{PHI3_INSTRUCTION}\n\nSegment:\n{text}<|end|>\n<|assistant|>"

    @classmethod
    def _clean_output(cls, text: str) -> str:
        """Trim generation at the first stop marker and strip surrounding whitespace."""
        cut = len(text)
        for marker in cls._STOP_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut].strip()

    def summarize_segment(self, segment_text: str) -> str:
        """Summarize a single segment with the Phi-3 ONNX model.

        Args:
            segment_text: The text to summarize

        Returns:
            Short third-person clinical summary.
        """
        og = self._og
        prompt = self._build_prompt(segment_text)
        # Only the tokenizer differs between backends; the og.Generator below
        # runs the model identically on whatever token IDs it is fed.
        if self.tokenizer_backend == "hf":
            input_tokens = self.hf_tokenizer.encode(prompt)
        else:
            input_tokens = self.og_tokenizer.encode(prompt)

        params = og.GeneratorParams(self.model)
        # Decoding is greedy and deterministic: do_sample=False (with num_beams=1)
        # means we always take the argmax token. Every generation parameter is set
        # explicitly here (rather than relying on genai_config defaults) so the
        # behaviour is self-documenting and cannot silently change if a config or
        # library default shifts. The sampling knobs (temperature, top_k, top_p)
        # are inert under greedy decoding, and the repetition controls are
        # disabled (repetition_penalty=1.0, no_repeat_ngram_size=0). max_new_tokens
        # is the only generation-length control, applied here via max_length.
        params.set_search_options(
            max_length=len(input_tokens) + self.max_new_tokens,
            batch_size=1,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)

        if self.tokenizer_backend == "hf":
            # Collect the generated IDs and batch-decode with HF. Keep special
            # tokens so _clean_output can still trim at markers like <|end|>.
            generated_ids = []
            while not generator.is_done():
                generator.generate_next_token()
                generated_ids.append(int(generator.get_next_tokens()[0]))
            text = self.hf_tokenizer.decode(generated_ids, skip_special_tokens=False)
        else:
            # Fresh decode stream per segment so detokenization state never leaks
            # between segments.
            stream = self.og_tokenizer.create_stream()
            generated = []
            while not generator.is_done():
                generator.generate_next_token()
                next_token = int(generator.get_next_tokens()[0])
                generated.append(stream.decode(next_token))
            text = "".join(generated)

        return self._clean_output(text)

    def summarize_segments(self, segments: List["Segment"]) -> List[str]:
        """Summarize multiple segments.

        Args:
            segments: List of Segment objects

        Returns:
            List of summaries
        """
        return [self.summarize_segment(s.get_text()) for s in segments]
