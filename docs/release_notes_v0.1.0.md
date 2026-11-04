# Release notes: v0.1.0

First complete version of the RAG-Based Intelligent Product Recommendation Agent.

## Included

- Offline pipeline: validate catalog, build documents, fit and embed with a built-in embedder,
  persist an atomic, checksummed FAISS artifact set.
- Online workflow: query parsing with structured controls and follow-ups, hard-constraint
  eligibility, exact retrieval over the eligible subset, evidence records, grounded
  deterministic explanations, validation and safe rendering.
- FastAPI service (`/health`, `/ready`, `/recommendations`) with stable error codes, a stateless
  design (client-held session) and bounded input and concurrency.
- Streamlit interface that talks to the API only.
- 8-product test fixture, 60-product synthetic demo catalog, 60-query frozen evaluation set,
  metrics, baselines, calibration tool and published raw reports.
- CI (pinned actions, read-only permissions) running a single local-reproducible check script.

## Verified

Automated tests (no network, no model download), CI on GitHub, a clean-clone run of the README
quick start including the API and UI, and an evaluation on the held-out split
(see `docs/evaluation.md`).

## Not verified / known limits

- The optional E5 embedding backend and the optional local-LLM explainer were **never run against
  real models**; they are tested only with mocks and are not used by default.
- The built-in embedder is lexical plus a curated vocabulary: no negation handling, lexical false
  friends, and unreliable abstention (so `MIN_SIMILARITY` is off).
- Data and evaluation set are synthetic and single-author; results are optimistic and carry no
  confidence intervals.
- Binds to a local interface; no authentication or rate-limiting design for public deployment.

## Run it

See the README quick start. Everything needed is installed from PyPI; nothing is downloaded at run time.
