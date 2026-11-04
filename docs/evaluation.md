# Evaluation

Everything below is generated from the raw reports in `reports/` by
`scripts/render_evaluation_md.py`; no number is typed by hand. Regenerate the reports with the
commands at the end.

> **Read this first.** The catalog is **synthetic** (60 laptops written for this project) and the
> 60 evaluation queries and their relevance labels were written by **one author** using explicit
> rules (no second human review). The built-in embedder's concept lexicon was written by the same
> author with the same vocabulary as the catalog, so results on queries that use that vocabulary
> are **optimistic**. These numbers show that the pipeline and the evaluation tooling work and how
> the methods compare on this small set. They are **not** evidence of accuracy on real shoppers,
> real catalogs or neural embeddings, and with 30 positive test queries they carry no confidence
> intervals.

## Setup

| Item | Value |
|---|---|
| Catalog | `data/demo/products.jsonl`, 60 synthetic products, sha256 `42a7c29e4095` |
| Evaluation set | `data/eval/queries.jsonl`, 60 queries: 20 development + 40 held-out test, sha256 `03a6aa6bf717` |
| Test labels | frozen; digest `483dd2d0b5df` is verified on every run |
| Embedder | `builtin-hashed-tfidf-v1` (hashed TF-IDF + curated concept lexicon; **not** neural), revision `idf-31d8131ea345e958` |
| Explanations | deterministic templates; no language model was used or evaluated |
| Retrieval | exact search over the eligible subset, `retrieval_k` = 10, up to 3 shown |
| Hardware | Linux-6.18.44-fc-v233-x86_64-with-glibc2.39, 4 CPUs, Python 3.11.17, faiss 1.15.1 |

Relevance grades are 2 (strong), 1 (partial), 0 (irrelevant), assigned by rules over catalog tags
and specifications that were written independently of the retriever. Recall@10 counts relevant
**eligible** products found in the top 10; MRR@3 and nDCG@3 use graded relevance (gain 2^grade-1,
discount log2(rank+1)). Queries with no relevant eligible product are excluded from the ranking
averages and counted separately.

## Order of events (no tuning on the test split)

1. The abstention cutoff was calibrated on the **development** split only
   (`evaluate --calibrate`): threshold **0.0603**, balanced accuracy
   1.00, separable on dev: True
   (best irrelevant score 0.019, lowest answerable score
   0.102). Only 2 irrelevant development queries
   existed, so this calibration is statistically weak.
2. The **test** split was then run once with that cutoff, and once without it as a reference. Both
   configurations were chosen before looking at test results; nothing was changed afterwards.
3. The cutoff did not generalize well enough to be enabled (see "Findings"), so
   `MIN_SIMILARITY` stays **unset by default**. This decision was made after seeing the test
   result, and the calibration command remains available.

## Retrieval quality (held-out test split)

Retrieval metrics are computed on the ground-truth eligible products so they isolate ranking from
constraint parsing, and they do not depend on the cutoff.

| Method | Recall@10 | MRR@3 | nDCG@3 |
|---|---|---|---|
| Application retriever (built-in embedder) | 0.721 | 0.983 | 0.921 |
| Keyword baseline (BM25) | 0.647 | 0.950 | 0.854 |
| Ablation: global top-10 filtered afterwards | 0.621 | 0.983 | 0.921 |
| Random order (floor) | 0.353 | 0.454 | 0.279 |
| Oracle ceiling for Recall@10 | 0.839 | - | - |

*n = 30 queries with at least one relevant eligible product.*

Development split, for reference:

| Method | Recall@10 | MRR@3 | nDCG@3 |
|---|---|---|---|
| Application retriever (built-in embedder) | 0.566 | 0.938 | 0.862 |
| Keyword baseline (BM25) | 0.499 | 1.000 | 0.874 |
| Ablation: global top-10 filtered afterwards | 0.493 | 0.938 | 0.862 |
| Random order (floor) | 0.274 | 0.543 | 0.309 |
| Oracle ceiling for Recall@10 | 0.608 | - | - |

*n = 16 queries with at least one relevant eligible product.*

How to read this:

- Filtering **before** searching beats searching globally and filtering afterwards (Recall@10
  0.721 vs 0.621 on test), which is why
  the design filters first.
- On the test split the built-in embedder is ahead of the BM25 baseline on all three metrics, but
  on the development split BM25 has the higher MRR@3 (1.000 vs
  0.938). With 16 and 30 queries the two methods should be considered
  **comparable**, not ranked.
- Recall@10 is capped by the number of relevant products: where more than 10 products are relevant
  the ceiling is below 1. Compare Recall@10 with the oracle ceiling row, not with 1.0.

### By query category (test)

| Category | n | builtin R@10 | builtin MRR@3 | builtin nDCG@3 | keyword R@10 | keyword MRR@3 | keyword nDCG@3 |
|---|---|---|---|---|---|---|---|
| constraints | 6 | 0.933 | 1.000 | 1.000 | 0.878 | 1.000 | 0.912 |
| direct | 6 | 0.644 | 1.000 | 0.985 | 0.532 | 1.000 | 0.985 |
| injection | 3 | 0.687 | 0.833 | 0.844 | 0.700 | 1.000 | 1.000 |
| model-name | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| negation | 2 | 0.461 | 1.000 | 0.667 | 0.176 | 0.500 | 0.128 |
| paraphrase | 6 | 0.466 | 1.000 | 0.849 | 0.365 | 0.917 | 0.755 |
| unknown-spec | 3 | 0.792 | 1.000 | 0.922 | 0.768 | 1.000 | 0.823 |

Per-category counts are tiny (2 to 6); treat them as illustrations, not estimates.

### Does the lexicon help? (test)

| Query set | n | builtin MRR@3 | builtin nDCG@3 | keyword MRR@3 | keyword nDCG@3 |
|---|---|---|---|---|---|
| uses words the concept lexicon knows | 23 | 0.978 | 0.936 | 0.935 | 0.837 |
| uses no word the lexicon knows | 7 | 1.000 | 0.871 | 1.000 | 0.913 |

The expected-lexicon flag is the author's own judgment of whether a query uses a word the curated
lexicon knows. Queries outside it are the fairer test of generalization, and there the built-in
embedder has no clear advantage over keyword search.

## End-to-end system

Through the full service (parsing, filtering, retrieval, evidence, grounding, rendering).

| Metric | Dev (no cutoff) | Test (cutoff) | Test (no cutoff) |
|---|---|---|---|
| Status accuracy (ok / no_match / needs_clarification) | 0.900 (18/20) | 0.975 (39/40) | 0.950 (38/40) |
| Constraint interpretation accuracy | 1.000 (17/17) | 1.000 (33/33) | 1.000 (33/33) |
| Constraint violation rate (returned products failing a hard constraint) | 0.000 (0/54) | 0.000 (0/87) | 0.000 (0/96) |
| Answer rate on answerable queries | 1.000 (16/16) | 1.000 (30/30) | 1.000 (30/30) |
| False-empty rate (no_match on an answerable query) | 0.000 (0/16) | 0.000 (0/30) | 0.000 (0/30) |
| No-match accuracy | 1.000 (1/1) | 1.000 (3/3) | 1.000 (3/3) |
| Clarification accuracy | 0.333 (1/3) | 0.857 (6/7) | 0.714 (5/7) |
| Abstention on irrelevant queries | 0.000 (0/2) | 0.500 (1/2) | 0.000 (0/2) |
| Evidence validity (displayed claims audited against the catalog) | 1.000 (756/756) | 1.000 (1218/1218) | 1.000 (1344/1344) |
| MRR@3 of the returned top 3 | 0.938 (n=16) | 0.983 (n=30) | 0.983 (n=30) |
| nDCG@3 of the returned top 3 | 0.862 (n=16) | 0.905 (n=30) | 0.921 (n=30) |

## Findings

**What holds up**

- **Hard constraints were never violated**: 0 of 87
  returned products on test (0 of 54 on dev), including queries with
  unknown required specifications and with instructions embedded in the query text.
- **Every displayed claim was valid**: 1218 of
  1218 claims on test matched the catalog under an audit that is independent of the
  grounding validator. This measures grounding against the catalog, not that the catalog is right
  or that a reason is *useful*.
- Constraints written in the query text or passed as controls were interpreted correctly on every
  evaluated query (1.000 (33/33)); ambiguous,
  contradictory and unsupported requests produced a question rather than a guess.

**What does not**

- **Abstention on irrelevant queries is unreliable.** With the calibrated cutoff
  0.0603, 0.500 (1/2) irrelevant test
  queries abstained; without a cutoff, 0.000 (0/2).
  `I4` ("best hiking trails near Denver") scored 0.137, **above** the
  lowest-scoring answerable query (P10: 0.066), so no raw-score cutoff separates
  them on this embedder.
- **The cutoff has a cost.** It cut the results of 2 answerable queries
  (P4, P10), both paraphrases that use no word the lexicon knows; the returned
  nDCG@3 fell from 0.921 to 0.905.
- **Negation is not handled.** "coding laptop that is not heavy" scored nDCG@3
  0.33 with the built-in embedder and 0.26 with BM25. On "office laptop
  without a dedicated graphics card" the built-in embedder scored 1.00 while BM25 scored
  0.00: keyword matching is pulled toward products that mention dedicated graphics, and the
  built-in embedder was not, most likely because "office" outweighed the rest. Neither method
  understands "without"; with two queries these are anecdotes, and the category averages
  (0.667 vs 0.128) should not be read as a measure of negation ability.
- **Instruction-like text lowers ranking quality.** On the three queries with embedded
  instructions, nDCG@3 was 0.844 for the built-in embedder against 1.000 for BM25. The
  likely cause is that the extra words add character n-gram features to the query; that
  explanation was not isolated or tested. This is a quality problem, not a safety one: no
  ineligible product was returned.
- **Queries that use no word the concept lexicon knows rank worse for the built-in embedder**
  (nDCG@3 0.871 against
  0.936 for queries that do), and on that subset
  BM25 is slightly ahead (0.913).

## Operational measurements (test run)

| Measure | Value |
|---|---|
| Warm end-to-end latency | p50 0.84 ms, p95 1.17 ms (800 warm requests) |
| Cold first request | 9.01 ms |
| Slowest stage (mean) | `explain` (0.353 ms); all stage means: parse 0.098, filter 0.027, retrieve 0.260, select 0.028, explain 0.353, render 0.084 ms |
| Catalog build (60 products) | 0.029 s |
| Peak memory | 69 MB resident (whole process) |
| Requests issued | 840 |
| Fallback rate | not applicable: no language model is used by default |

Latency is for in-process calls on the hardware above with a 60-product catalog. It is not a
claim about larger catalogs, concurrency or a networked deployment.

## What these numbers do not show

- Accuracy on real users, real product data or real catalogs.
- Quality of a neural embedder or of a language-model explainer: neither was run here.
- Statistical significance: there are no confidence intervals and the per-category groups are tiny.
- Label reliability: a single author wrote queries, rules and catalog; there was no second review.
- Generalization beyond this vocabulary: the concept lexicon and the catalog share an author.

## Reproduce

```bash
python -m recommender.cli evaluate --calibrate --runs 20 --output reports/eval-dev.json
python -m recommender.cli evaluate --split test --min-score 0.0603 --runs 20 --output reports/eval-test.json
python -m recommender.cli evaluate --split test --runs 20 --output reports/eval-test-no-cutoff.json
python scripts/render_evaluation_md.py
```

The evaluation reads neither `.env` nor environment variables, refuses to run if the catalog or the
frozen test labels changed, and writes per-query rankings and outputs next to the aggregates in
each report.
