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
