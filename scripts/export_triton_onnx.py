#!/usr/bin/env python3
"""Export a Hugging Face sequence classifier to ONNX for Triton."""

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import onnx
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'onnx'. Install with: conda run -n chirp pip install onnx"
    ) from exc

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'onnxruntime'. Install with: conda run -n chirp pip install onnxruntime"
    ) from exc


ORT_TO_TRITON_DTYPE = {
    "tensor(bool)": "TYPE_BOOL",
    "tensor(double)": "TYPE_FP64",
    "tensor(float)": "TYPE_FP32",
    "tensor(float16)": "TYPE_FP16",
    "tensor(int16)": "TYPE_INT16",
    "tensor(int32)": "TYPE_INT32",
    "tensor(int64)": "TYPE_INT64",
    "tensor(int8)": "TYPE_INT8",
    "tensor(string)": "TYPE_STRING",
    "tensor(uint16)": "TYPE_UINT16",
    "tensor(uint32)": "TYPE_UINT32",
    "tensor(uint64)": "TYPE_UINT64",
    "tensor(uint8)": "TYPE_UINT8",
}


class SequenceClassifierExportWrapper(torch.nn.Module):
    """Wrap a HF classifier to expose logits only."""

    def __init__(self, model: torch.nn.Module, input_names: Sequence[str]):
        """Store model and fixed input name ordering for ONNX export.

        Args:
            model: Hugging Face sequence classification model.
            input_names: Ordered list of input tensor names used to map positional
                ONNX arguments back to keyword arguments.
        """
        super().__init__()
        self.model = model
        self.input_names = list(input_names)

    def forward(self, *model_inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns logits only.

        Args:
            *model_inputs: Positional tensors in `self.input_names` order.

        Returns:
            Logits tensor from the wrapped classifier.
        """
        inputs = {name: value for name, value in zip(self.input_names, model_inputs)}
        return self.model(**inputs).logits


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face classifier to Triton ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing a Hugging Face model",
    )
    parser.add_argument(
        "--triton-repo",
        type=Path,
        required=True,
        help="Target Triton model repository root directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="chirpe_classifier",
        help="Model name inside Triton model repository",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Numeric Triton model version directory",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Triton max_batch_size",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max tokenizer length for dummy export inputs",
    )
    parser.add_argument(
        "--dummy-text",
        type=str,
        default="The participant reports unusual thoughts and occasional confusion.",
        help="Seed text used to build dummy export inputs",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="export_metadata.json",
        help="Metadata filename saved next to model.onnx",
    )
    return parser.parse_args()


def get_dummy_inputs(
    model,
    tokenizer,
    dummy_text: str,
    max_length: int,
) -> Tuple[List[str], Tuple[torch.Tensor, ...]]:
    """Build sample inputs used to trace/export the ONNX graph.

    Args:
        model: Hugging Face classifier model.
        tokenizer: Hugging Face tokenizer associated with the model.
        dummy_text: Seed sentence used for dummy tokenization.
        max_length: Tokenizer maximum sequence length.

    Returns:
        Tuple of ordered input names and corresponding input tensors.

    Raises:
        ValueError: If `input_ids` is missing from resolved model inputs.
    """
    batch = tokenizer(
        [dummy_text, f"{dummy_text} Additional context for dynamic sequence axes."],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    accepted_names = set(inspect.signature(model.forward).parameters)
    input_names = [
        name
        for name in tokenizer.model_input_names
        if name in batch and name in accepted_names
    ]
    if "input_ids" not in input_names:
        raise ValueError("Tokenizer inputs do not include input_ids; unsupported model layout.")

    tensors = tuple(batch[name] for name in input_names)
    return input_names, tensors


def make_dynamic_axes(
    input_names: Sequence[str],
    input_tensors: Sequence[torch.Tensor],
) -> Dict[str, Dict[int, str]]:
    """Create dynamic axis metadata for ONNX export.

    Batch dimension is always dynamic; sequence length is dynamic for rank>=2
    input tensors.
    """
    dynamic_axes: Dict[str, Dict[int, str]] = {}

    for name, tensor in zip(input_names, input_tensors):
        axes = {0: "batch_size"}
        if tensor.ndim >= 2:
            axes[1] = "sequence_length"
        dynamic_axes[name] = axes

    dynamic_axes["logits"] = {0: "batch_size"}
    return dynamic_axes


def export_model(
    wrapper: torch.nn.Module,
    model_path: Path,
    input_names: Sequence[str],
    input_tensors: Sequence[torch.Tensor],
    opset: int,
) -> None:
    """Export wrapper model to ONNX with dynamic axes.

    Args:
        wrapper: Export wrapper that outputs logits.
        model_path: Output path for `model.onnx`.
        input_names: ONNX input names in positional order.
        input_tensors: Dummy tensors matching `input_names`.
        opset: ONNX opset version.
    """
    dynamic_axes = make_dynamic_axes(input_names, input_tensors)

    torch.onnx.export(
        wrapper,
        args=tuple(input_tensors),
        f=str(model_path),
        input_names=list(input_names),
        output_names=["logits"],
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        external_data=False,
    )


def triton_dims(shape: Sequence, max_batch_size: int) -> List[int]:
    """Convert ONNX tensor shape to Triton dims format.

    Args:
        shape: ONNX Runtime shape metadata.
        max_batch_size: Triton max batch size from config.

    Returns:
        Dimension list excluding batch axis when batching is enabled.
    """
    start = 1 if max_batch_size > 0 else 0
    dims: List[int] = []
    for dim in list(shape)[start:]:
        if isinstance(dim, int):
            dims.append(dim if dim >= 0 else -1)
        else:
            dims.append(-1)
    return dims


def format_dims(dims: Sequence[int]) -> str:
    """Format dimension list for Triton `config.pbtxt` emission."""
    if not dims:
        return ""
    return ", ".join(str(dim) for dim in dims)


def to_triton_dtype(ort_type: str) -> str:
    """Map ONNX Runtime dtype string to Triton dtype enum token."""
    if ort_type not in ORT_TO_TRITON_DTYPE:
        raise ValueError(f"Unsupported ORT tensor type for Triton config: {ort_type}")
    return ORT_TO_TRITON_DTYPE[ort_type]


def build_config_pbtxt(
    model_name: str,
    max_batch_size: int,
    onnx_inputs,
    onnx_outputs,
) -> str:
    """Generate Triton `config.pbtxt` content from ONNX I/O metadata."""
    lines = [
        f'name: "{model_name}"',
        'backend: "onnxruntime"',
        f"max_batch_size: {max_batch_size}",
        "",
        "input [",
    ]

    for model_input in onnx_inputs:
        dtype = to_triton_dtype(model_input.type)
        dims = format_dims(triton_dims(model_input.shape, max_batch_size))
        lines.append(
            f'  {{ name: "{model_input.name}" data_type: {dtype} dims: [ {dims} ] }}'
        )

    lines.extend(["]", "", "output ["])

    for model_output in onnx_outputs:
        dtype = to_triton_dtype(model_output.type)
        dims = format_dims(triton_dims(model_output.shape, max_batch_size))
        lines.append(
            f'  {{ name: "{model_output.name}" data_type: {dtype} dims: [ {dims} ] }}'
        )

    lines.extend(
        [
            "]",
            "",
            "instance_group [",
            "  { kind: KIND_CPU }",
            "]",
        ]
    )

    if max_batch_size > 0:
        lines.extend(["", "dynamic_batching {}"])

    return "\n".join(lines) + "\n"


def save_metadata(
    metadata_path: Path,
    source_model_dir: Path,
    onnx_model_path: Path,
    onnx_inputs,
    onnx_outputs,
) -> None:
    """Persist export metadata JSON next to the ONNX model.

    The metadata file helps downstream tooling introspect input/output names,
    shapes, and original model provenance.
    """
    payload = {
        "source_model_dir": str(source_model_dir),
        "onnx_model_path": str(onnx_model_path),
        "inputs": [
            {"name": item.name, "type": item.type, "shape": list(item.shape)}
            for item in onnx_inputs
        ],
        "outputs": [
            {"name": item.name, "type": item.type, "shape": list(item.shape)}
            for item in onnx_outputs
        ],
    }
    with open(metadata_path, "w") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    """Run the ONNX export pipeline end-to-end.

    This validates arguments, loads the source model/tokenizer, exports ONNX,
    validates the graph, generates Triton config, and writes metadata.
    """
    args = parse_args()

    major_version = int(transformers.__version__.split(".")[0])
    if major_version >= 5:
        raise SystemExit(
            "ONNX export currently requires transformers<5 in this script. "
            "Install a 4.x release (for example: pip install 'transformers<5')."
        )

    if args.version <= 0:
        raise SystemExit("--version must be a positive integer")
    if args.max_batch_size < 0:
        raise SystemExit("--max-batch-size must be >= 0")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be > 0")
    if not args.model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {args.model_dir}")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir,
            attn_implementation="eager",
        )
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.eval()

    input_names, input_tensors = get_dummy_inputs(
        model=model,
        tokenizer=tokenizer,
        dummy_text=args.dummy_text,
        max_length=args.max_length,
    )

    wrapper = SequenceClassifierExportWrapper(model=model, input_names=input_names)
    wrapper.eval()

    model_root = args.triton_repo / args.model_name
    version_dir = model_root / str(args.version)
    version_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = version_dir / "model.onnx"
    config_path = model_root / "config.pbtxt"
    metadata_path = version_dir / args.metadata_file

    export_model(
        wrapper=wrapper,
        model_path=onnx_path,
        input_names=input_names,
        input_tensors=input_tensors,
        opset=args.opset,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = ort_session.get_inputs()
    ort_outputs = ort_session.get_outputs()

    config_text = build_config_pbtxt(
        model_name=args.model_name,
        max_batch_size=args.max_batch_size,
        onnx_inputs=ort_inputs,
        onnx_outputs=ort_outputs,
    )

    with open(config_path, "w") as file:
        file.write(config_text)

    save_metadata(
        metadata_path=metadata_path,
        source_model_dir=args.model_dir,
        onnx_model_path=onnx_path,
        onnx_inputs=ort_inputs,
        onnx_outputs=ort_outputs,
    )

    print(f"Export complete: {onnx_path}")
    print(f"Triton config written: {config_path}")
    print(f"Metadata written: {metadata_path}")
    print(f"ONNX inputs: {[item.name for item in ort_inputs]}")
    print(f"ONNX outputs: {[item.name for item in ort_outputs]}")


if __name__ == "__main__":
    main()
