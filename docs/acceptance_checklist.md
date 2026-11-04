# Final acceptance checklist

Each item from the design's release gate, with the evidence for it. Status: **verified** (checked
by a test, a measurement or a run described here), **partial**, or **not done**. Nothing is ticked
without evidence.

## Functionality

| Item | Status | Evidence |
|---|---|---|
| Normal requests return distinct eligible products with verified prices, specs and evidence references | verified | integration and API tests; held-out evaluation: 0/96 constraint violations, 1344/1344 displayed claims matched the catalog under an independent audit |
| Budget, RAM, weight and stock work at boundaries; unknown required fields are excluded; fewer than three matches stay fewer than three | verified | `tests/unit/test_catalog.py` (inclusive boundaries, unknown weight), `test_fewer_than_requested_matches_return_fewer` |
| Ambiguous requests ask a question; no-match stays empty; follow-ups and reset behave correctly | verified | service, API and Streamlit `AppTest` tests; evaluation: every ambiguous or contradictory query produced a question |
| LLM failures and invalid plans use deterministic explanations with a visible mode indicator | partial | verified with a **mocked** LLM (timeout, connection, malformed JSON, invented ID, wrong evidence); the UI shows "Basic explanation". The real local LLM was **never run** |

## Integrity and quality

| Item | Status | Evidence |
|---|---|---|
| Catalog, embedder revision, vector normalization, ID mapping and manifest agree after save/load | verified | `tests/unit/test_index.py`, `tests/integration/test_offline_pipeline.py` (shuffled mapping, tampering, mismatched embedder) |
| Unit and integration checks and CI pass | verified | CI run on GitHub for the release commit; the same `scripts/check.sh` was also run from scratch in a fresh virtual environment |
| Real-model checks are recorded separately | **not done** | the optional E5 backend and the optional Ollama explainer were never run against real models (the build environment could not download or host them). The default pipeline contains neither |
| No untrusted generated fact is rendered | verified | sentences come only from templates; the grounding validator and an independent audit; injection fixtures |
| Evaluation reports use frozen splits, explicit denominators, measured values, baselines and hardware context | verified | `docs/evaluation.md`, `reports/` (BM25, random and post-filter baselines; frozen test-label digest) |

## Reproducibility and publication

| Item | Status | Evidence |
|---|---|---|
| A clean clone can install dependencies, validate the data, rebuild artifacts and run the API and UI with the documented commands | verified | README quick start executed verbatim in a fresh clone (exit code 0); API and Streamlit health checked from that clone |
| README, screenshots, architecture, data card, code and data licensing and limitations are present; synthetic data and snapshot prices are labeled | verified | `README.md`, `docs/images/`, `docs/architecture.md`, `docs/data_card.md`, `LICENSE`, `DATA_LICENSE.md` |
| No secrets, virtual environments, private records or model weights are tracked | verified | repository scan before release: 83 tracked files, about 1 MB, none of the forbidden types, no secret-like strings (the only match is a test sentinel) |
| Repository identity and visibility are intentional | **needs your confirmation** | the repository is currently **public**; the maintainer should confirm that is intended |
| The repository contains the tested release and its CI result; a release tag and short release notes identify the scope | partial | CI passed on GitHub for every commit up to the release commit, and `docs/release_notes_v0.1.0.md` identifies the scope. The tag `v0.1.0` (and later `v0.2.0` for the Persian prototype) and a GitHub Release page could not be created from the build environment: pushing a tag from the build environment was rejected with HTTP 403 (branch pushes work). See "Publishing the tag" below |

## Persian track (post-v0.1.0)

| Item | Status | Evidence |
|---|---|---|
| Persian constraints are enforced and toman is asked about | verified | `reports/fa/` (0 of 96 violations, 33 of 33 readings correct, against 22 of 120 before), `tests/unit/test_query_fa.py` |
| English results unchanged by the Persian work | verified | regression test comparing a fresh run with the published report |
| Persian query texts reviewed by a native speaker | **not done** | `docs/fa_review_sheet.md` is ready for a reviewer; until then this is a prototype |
| Persian responses | **not done** | out of scope: responses are English |

## What this release deliberately is not

- Not evidence of accuracy on real shoppers, real catalogs, neural embeddings or a language model.
- Not a deployed service: it binds to a local interface and has no authentication or rate limiting design.
- Not statistically validated: 30 positive held-out queries, no confidence intervals, one author.

## Publishing the tag

Tag pushes are blocked in the environment this project was built in, so the owner creates the tag
once, on the commit whose CI run is green (the latest commit on `main` at the time of release):

```bash
git fetch origin && git checkout main && git pull
git tag -a v0.1.0 -m "v0.1.0: first complete version (see docs/release_notes_v0.1.0.md)"
git push origin v0.1.0
```

Or in the GitHub web interface: Releases, "Draft a new release", create the tag `v0.1.0` targeting
`main`, and paste the contents of `docs/release_notes_v0.1.0.md` as the description.
