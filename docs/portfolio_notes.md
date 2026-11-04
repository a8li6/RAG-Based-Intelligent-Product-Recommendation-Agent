# Portfolio notes

How to present this project honestly. Every number below is taken from `reports/` (held-out test
split, shipped default configuration) and is checked against them by a test.

## What to claim, and what not to

**Accurate:** a Python recommendation application that enforces budget, RAM, weight and stock
constraints in code, retrieves with an exact FAISS index over the eligible products, and renders
only catalog-verified, evidence-backed explanations, with an API, a UI, an evaluation harness and CI.

**Do not claim:** neural or multilingual embeddings, a language model generating the answers,
real-user accuracy, or production readiness. The default pipeline has none of the first two, the
optional adapters were never run against real models, and the data is synthetic.

## Resume bullets (measured, with context)

- Built a Python product-recommendation application (FAISS exact retrieval, FastAPI, Streamlit) that
  enforces budget, RAM, weight and stock constraints in code and shows only catalog-verified,
  evidence-backed explanations, with a deterministic fallback and stable API errors.
- Evaluated on a synthetic 60-product catalog with a frozen 40-query held-out set: 0/96 constraint
  violations in returned products, 1344/1344 displayed claims matching the catalog under an independent audit, and
  filter-first retrieval reaching Recall@10 0.72 versus 0.62 for post-filtering and 0.65 for BM25
  (small, single-author evaluation).
- Delivered reproducible engineering around it: pinned dependencies, atomic checksummed index
  artifacts, over 600 automated tests, and a GitHub Actions workflow that runs the same script as local checks.

## 90-120 second demo

1. One sentence: filters understand money but not meaning, chatbots understand meaning but invent facts; this combines them.
2. Open the UI, point at the "Synthetic demo catalog / snapshot prices" label, type
   "Portable laptop for coding and commuting" with budget 900, RAM 16, weight 1.5.
3. Show the active filters, then expand "Evidence and sources" on a card.
4. Type "under $700 instead": the budget changes and RAM and weight are preserved.
5. Type "something cheap": it asks for a budget instead of inventing one. Try a budget under $300 for the no-match explanation.
6. Stop the API and submit again to show the plain-language "service not reachable" message.
7. Finish on `docs/evaluation.md`: what was measured, and the limitations section.

## Explaining the design in an interview

- **Why filter before searching?** Filtering a global top-10 afterwards can lose the only eligible
  product. Measured: Recall@10 0.72 versus 0.62.
- **Why are vectors normalized and price kept out of the embedding?** With unit vectors the inner
  product equals cosine similarity, and volatile values enforced by exact code never force re-embedding.
- **Why versioned, checksummed artifacts?** A mismatched ID mapping silently returns the wrong
  product; the loader detects shuffled mappings (even with a rewritten checksum) before serving.
- **How is the model bounded?** The planner can only pick product IDs, evidence IDs and reason
  codes; sentences come from templates, and a validator rejects anything unsupported.
- **How did you separate retrieval quality from explanation quality?** Retrieval metrics never see
  explanations, and baselines (BM25, random, post-filter) use the same eligible products.

## What went wrong, and what I learned (honest stories)

- The abstention cutoff was calibrated on the development split and **did not generalize**: an
  irrelevant query scored above an answerable one. I kept it off by default and documented it
  instead of tuning on the test split.
- Tests caught real bugs: "no more than $900" was parsed as a minimum price; environment variables
  could silently change evaluation results; a duplicated query text leaked across splits.
- A generated-document script asserts each claim against the data; it caught three statements in my
  own first draft that the numbers did not support.
- The final pipeline is offline and lexical, not neural, by choice: it removes every network
  dependency, and the evaluation shows the cost (no negation handling, lexical false friends).
