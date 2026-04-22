# String-In ONNX Artifact Specification

Date: 2026-04-22

## Goal
Define the exact model artifacts for the string-input ONNX migration.

## Required Artifacts (One per Backbone)
1. `chirpe_bert_string`
2. `chirpe_clinicalbert_string`
3. `chirpe_mentalbert_string`

Each artifact is a single fused ONNX graph:
- Input: raw text (`TYPE_STRING`)
- Output: classifier `logits`

## Inference Contract
- One backbone is selected per run.
- Transcript prediction remains segment-level classification followed by vote aggregation.
- No cross-backbone ensemble inference.

## Triton Repository Layout (Expected)
For each model name above:

```
<triton_repo>/<model_name>/
  config.pbtxt
  1/
    model.onnx
    export_metadata.json
```

## Acceptance Mapping
- Satisfies checklist item: "Confirm target artifacts: 3 fused ONNX models (`bert`, `clinicalbert`, `mentalbert`)."
