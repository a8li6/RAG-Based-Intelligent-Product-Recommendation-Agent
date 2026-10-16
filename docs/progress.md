# Progress Log

## Session 1 - M0 and M1 (catalog and filters)

**Completed**
- `.gitignore`, README stub, `docs/product_brief.md`.
- Packaging (`pyproject.toml`), package skeleton, `.env.example`.
- `schemas.py`: strict `Product` and `Constraints` contracts.
- `data/sample/products.jsonl`: eight synthetic fixture products.
- `catalog.py`: line-numbered validation, unique IDs, canonical snapshot, SHA-256,
  hard-constraint eligibility.
- `documents.py`: deterministic search documents (price/stock excluded).
- `cli.py`: `validate-data` command.
- 56 unit tests, passing; Ruff clean.

**Verified invariants**
- Budget 90,000 cents, RAM >= 16 GB, weight <= 1.5 kg, in stock: eligible = {P001, P004}.
- Budget 70,000, RAM >= 16: {P004, P008}; adding weight <= 1.5 leaves {P004}.
- Budget 30,000, RAM >= 16: no eligible product.

**Known issues / deviations**
- Heavy dependencies (faiss-cpu, sentence-transformers) are listed in
  `pyproject.toml` but not yet installed or pinned; `requirements-lock.txt` is
  deferred until the M2 embedding pilot passes.
- LICENSE (MIT) and DATA_LICENSE.md added after session 1's M1 commits.
- Ollama is not available in the cloud session; local-LLM checks are deferred.


## Session 2 - M2 (embeddings and retrieval)

**Completed**
- `config.py`: validated settings; paths resolve from the project root.
- `embeddings.py`: E5 adapter (passage/query prefixes, normalization, lazy model load)
  and `validate_vectors` (2D, float32, finite, non-zero, unit-norm).
- `index.py`: atomic artifact build (`products.faiss`, `embeddings.npy`,
  `product_ids.json`, `catalog.jsonl`, `manifest.json` with per-file checksums);
  loading rejects missing files, checksum mismatch, shuffled ID mapping (including
  a rewritten checksum), index/vector disagreement and embedder mismatch.
- `retrieval.py`: exact search over the eligible subset only. A test proves the
  only eligible product ranked below the global top-10 is still returned.
- CLI: `build-index` and `search` (embedder injectable for tests).
- 117 unit tests passing (`-m "not slow"`); Ruff clean.

**Not yet verified (real model)**
- The real `intfloat/multilingual-e5-small` model has **not** been run. The cloud
  environment's network policy blocks `huggingface.co`, so every M2 test uses a
  deterministic hash embedder. Consequences:
  - the M0 embedding pilot (10 records, peak memory, timing) is still open;
  - `EMBEDDING_REVISION` is still `main`; it must be resolved to an immutable
    commit hash before any benchmark or release;
  - `requirements-lock.txt` is not created; `sentence-transformers` is not installed;
  - semantic quality of the real encoder is unmeasured. The test embedder only
    proves plumbing, not relevance.
- A `slow`-marked test (`test_real_e5_model_roundtrip`) is ready for a machine
  with model access: `python -m pytest -m slow`.

**Next:** M3 - query parsing and follow-up rules, ranking and evidence, LLM adapter,
grounding validation, deterministic fallback, shared service.

## Session 3 - M3 (query, generation, grounding, service)

**Completed**
- `query.py`: narrow English patterns for budget / minimum RAM / maximum weight / in-stock,
  parsed with `Decimal`; structured `Filters` are authoritative; session follow-ups
  ("under $700 instead", "forget the weight limit"); ambiguity becomes a question.
- `ranking.py`: deduplicated, ID-tie-broken selection (ineligible candidates dropped even if
  retrieved); immutable evidence records (`E_P001_WEIGHT`) with per-record allowed reason codes.
- `grounding.py`: plan parsing and validation (IDs, order, evidence ownership, allowed codes,
  duplicates, false "unknown" claims, tampered context, hard-constraint recheck);
  deterministic plan; rendering only from catalog values and approved templates.
- `llm.py`: versioned prompt `explanation-plan-v1`, escaped untrusted text, one non-streaming
  Ollama request, stable failure codes, no retry.
- `service.py`: one `recommend()` shared by CLI and (later) API; statuses `ok` / `no_match` /
  `needs_clarification`; deterministic fallback with a bounded `fallback_reason`; no raw user
  text in logs.
- CLI: `recommend` prints the public response contract.
- 324 tests passing (unit + integration, `-m "not slow"` equivalent); Ruff clean.

**Design decisions (documented, not hidden)**
- "under $900" -> 89,999 cents; "at most $900" -> 90,000 cents.
- Weight limits are inclusive: "under 1.5 kg" is treated as "at most 1.5 kg".
- A bare "16 GB RAM" is a soft preference; only "at least 16 GB RAM" (and similar) is hard.
- A stray amount without a recognizable cue ("laptop for $900") asks a question instead of
  guessing a budget.
- Use-case reasons need exact word matches ("coding" matches the use case "coding"; "code"
  does not). No stemming.
- The abstention cutoff (`min_score`) exists but is **off**: it must be calibrated on the
  development set in M5.

**Bug found by tests during this session:** "no more than $900" was parsed as a minimum
price because of the substring "more than". Fixed for price and weight, with regression tests.

**Not yet verified with real components**
- The real `qwen3:4b` model via Ollama has **not** been run: Ollama is not available in this
  cloud environment. Request/response handling is verified only against `httpx.MockTransport`.
  Unconfirmed assumptions: the installed Ollama/model accepts `think: false` and a JSON-schema
  `format`; real model output quality and latency are unmeasured. The adapter treats any
  rejection as a fallback, so the service stays correct either way.
- The real E5 embedding model is still not run (Hugging Face blocked), so semantic ranking
  quality is unmeasured; tests use a hash embedder.
- Public error codes for internal failures (guide: "convert internal errors into stable public
  error codes") are an API concern and are deferred to M4.

**Next:** M4 - FastAPI (`/health`, `/ready`, `/recommendations`), Streamlit UI, API tests
with `TestClient`.

## Audit after M3 and gap closure (session 3)

An audit against the design guide found M0-M3 code complete but M4 (API and UI) not started,
plus several small M0/M1 gaps. The gaps that do not need network access are now closed.

**Closed**
- `tests/fixtures/`: invalid-catalog and hostile-product fixtures, with tests that assert every
  error is reported with its line and that injection text stays inert end to end.
- Validation report: `validate-data --report` (counts, catalog hash, missing-field counts,
  transformation rules, size and SHA-256 of the untouched source) and `--canonical-out`
  (deterministic snapshot). Outputs can never be the input file; nothing is written on failure.
- Build timing (`embed_seconds`, `build_seconds`) and `revision_pinned` recorded in the manifest;
  `build-index` warns when the embedding revision is not an immutable commit hash.
- Document token-length check using the model tokenizer (passage prefix included). Overlong
  documents are rejected before embedding, never truncated; token statistics go in the manifest.
- 345 tests passing (1 slow test skipped); Ruff clean.

**Still open - blocked by the cloud environment's network policy**
- Run the real `intfloat/multilingual-e5-small` model (`huggingface.co` is not reachable):
  the 10-record embedding pilot, peak memory and timing, and resolving `EMBEDDING_REVISION`
  from `main` to a commit hash.
- `requirements-lock.txt` (needs the full dependency set installed and verified, including the
  real model path) and the clean-environment install check that depends on it.
- Run the real local LLM (Ollama is not available): confirm `think: false` and JSON-schema
  `format` support, record the model digest and Ollama version, measure latency and quality.
- The token-length check is verified only with a stand-in tokenizer; the real tokenizer path
  (`E5Embedder.count_tokens`) is untested until the model can be loaded.

**Not started**
- M4: `api.py`, `ui/app.py`, API tests with `TestClient`.
- M5: reviewed 40-100 record catalog, 60-query evaluation set, `evaluate` command, metrics.
- M6: CI workflow, `docs/architecture.md`, `docs/evaluation.md`, full README, release.

**Note for M5:** with the 8-product fixture, Recall@10 is trivial. M5 first needs the larger
reviewed catalog. Retrieval-quality numbers computed with the hash test embedder are not
meaningful; they only validate the evaluation tooling until the real model can run.
