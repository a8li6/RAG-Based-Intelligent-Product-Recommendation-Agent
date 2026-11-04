# Environment

The project runs fully offline after dependencies are installed from PyPI. It needs no model
download, no hosted API and no local model server.

## Verified environment

| Item | Value |
|---|---|
| Python | 3.11.17 (project requires >=3.11,<3.12) |
| OS / architecture | Linux 6.18.44-fc-v233 / x86_64 |
| Lock file | `requirements-lock.txt` (51 pinned packages), produced with `pip freeze --exclude-editable` |
| Embedder | `builtin-hashed-tfidf-v1` (no weights; fitted state is stored with each index) |
| Embedder revision on the sample catalog | `idf-ee0205dba0bca2e4` (a content hash of the fitted state) |

A lock file from one machine does not establish cross-platform compatibility. If installation
fails on another OS or CPU architecture, install from `pyproject.toml` instead and re-freeze
only after the tests pass.

## Reproduce from a clean clone

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest -q
```

This sequence was run in a fresh virtual environment: `pip check` was clean and all tests passed.

## Optional extras (not required, not verified here)

- `pip install -e ".[neural]"` adds `sentence-transformers` for the optional E5 backend
  (`EMBEDDING_BACKEND=e5`). It needs a one-time model download and was **not** run in this
  environment, which cannot reach Hugging Face.
- `EXPLANATION_MODE=llm` enables the optional local-LLM explainer (Ollama). It is exercised
  only against mocked HTTP responses and was **not** run against a real model.
