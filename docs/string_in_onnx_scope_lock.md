# String-In ONNX Scope Lock

Date: 2026-04-22

## Confirmed Scope
- CHiRPE inference remains single-model per run.
- Supported model backbones are `bert`, `clinicalbert`, and `mentalbert`.
- Segment-level transcript decision remains vote aggregation across segments.

## Explicitly Out of Scope
- Multi-model ensemble inference in a single prediction path.
- Combined voting across BERT + ClinicalBERT + MentalBERT in one run.

## Rationale
- Matches current CLI and runtime flow.
- Keeps deployment and validation focused on tokenizer+classifier fusion per backbone.
- Avoids introducing unused/ambiguous inference paths.

## Acceptance Mapping
- Satisfies checklist item: "Confirm scope: single-model per run + segment voting (no multi-model ensemble)."
