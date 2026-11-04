# Changelog

## Unreleased (planned v0.2.0): Persian query prototype

- Persian query understanding over the English catalog: digit and separator normalization, constraint
  phrases, currency policy (toman is asked about, not guessed), a concept lexicon and a translation table to the
  catalog's vocabulary. Responses stay English.
- Separate Persian evaluation (`data/eval/fa/`, `reports/fa/`, `docs/evaluation_fa.md`) with a
  before/after comparison and a native-speaker review sheet.
- Matched-pair latency measurement; CI smoke-tests the Persian evaluation.
- Status: prototype. The Persian queries were AI-written and are **not reviewed by a native speaker**.

## v0.1.0

First complete version: offline hard-constraint-filtered retrieval, grounded deterministic explanations,
FastAPI service, Streamlit UI, frozen English evaluation set and reports, CI. See `docs/release_notes_v0.1.0.md`.
