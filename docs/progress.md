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
- No LICENSE file yet: the license choice is pending.
- Ollama is not available in the cloud session; local-LLM checks are deferred.

**Next:** M2 - embeddings adapter, FAISS index with manifest, eligible-subset retrieval.
