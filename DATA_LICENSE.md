# Data and Model Licensing

The project **source code** is licensed under the MIT License (see `LICENSE`).
A code license does not grant rights to third-party data or model weights.

## Sample data

`data/sample/products.jsonl` is original, fully synthetic data written for this
project. It is not derived from any third-party catalog and is distributed under
the same MIT terms as the code.

## Third-party data

Any real catalog added later must record its origin, access date, license and
redistribution permission in `docs/data_card.md` before being committed. Its
license is independent of the project's code license.

## Models

Model weights are **not** stored in this repository. Each model is downloaded
separately and is governed by its own license; check it before redistribution:

- Embedding model: `intfloat/multilingual-e5-small` (see its model card).
- Local generation model: `qwen3:4b` via Ollama (see its Ollama library page).
