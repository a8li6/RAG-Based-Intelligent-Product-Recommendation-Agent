# Persian (Farsi) evaluation

Generated from the raw reports in `reports/fa/` by `scripts/render_evaluation_fa_md.py`; no number
is typed by hand. This document is separate from `docs/evaluation.md` on purpose: English and Persian
results are never merged.

> **Read this first.**
> - The Persian query texts were **written by an AI assistant, not by native speakers, and have not
>   been reviewed by a native speaker**. Some phrasing may be unnatural, and real users will write
>   differently (colloquial spelling, typos, transliterated Persian in Latin letters).
> - Relevance labels are **rule-based on the English catalog's tags**, identical to the English set.
> - The Persian lexicon and translation table were written by the **same author** as the queries, so
>   results are optimistic, in the same way as the English ones.
> - **Scope:** Persian *query understanding* over the English catalog. Responses, reasons and
>   questions stay **English**. The catalog is not translated.
> - 60-product synthetic catalog, 40 held-out test queries, no confidence intervals.

## What was built

| Piece | What it does |
|---|---|
| `persian.py` normalization | Persian and Arabic digits, the Persian decimal (`٫`) and thousands (`٬`) separators, the Persian comma, slash decimals before "kilo", Arabic yeh/kaf, zero-width non-joiner |
| Persian constraint phrases | budgets (زیر، کمتر از، حداکثر، تا، بودجه، «یا کمتر»), RAM minimums, weight limits, in-stock, clearing a limit, and the same clarification questions as English |
| Currency policy | only USD; amounts in **toman**, rial, euro, pound are asked about, never guessed or silently ignored |
| Concept lexicon + translation | Persian words map to the same concept ids as the English lexicon and are translated to the catalog's own use-case vocabulary (every catalog tag has Persian phrases). Unknown Persian words add no retrieval features |

## Order of events

1. The Persian set (60 queries, parallel to the English ones) was written and its test labels frozen
   by digest `5a5c688a7c34` (commit `a34ef91`).
2. **Baseline** with the unchanged system was measured (commit `c2d5ee0`).
3. Persian support was implemented (commits `2fdf851`, `1ee7565`, `9884a59`). The development split was
   run once with the finished support and **no change was made after seeing it**. Changes made while
   implementing came from my own unit and fixture tests, for example the case of Latin `RAM`/`USD`, the
   missing "lighter than" phrase, and a travel query that ranked differently from its English twin
   on the 8-product fixture, not from evaluation results.
4. The **test split was run once**, at commit `51211c0`. Nothing was changed afterwards.

## Before and after Persian support

The baseline is the same system before any Persian support existed.

| Metric | Test: before | Test: after | Dev: before | Dev: after |
|---|---|---|---|---|
| Constraint violations among returned products | 0.183 (22/120) | 0.000 (0/96) | 0.100 (6/60) | 0.000 (0/54) |
| Constraint interpretation accuracy | 0.667 (22/33) | 1.000 (33/33) | 0.824 (14/17) | 1.000 (17/17) |
| Status accuracy | 0.750 (30/40) | 0.950 (38/40) | 0.800 (16/20) | 0.900 (18/20) |
| No-match accuracy | 0.000 (0/3) | 1.000 (3/3) | 0.000 (0/1) | 1.000 (1/1) |
| Clarification accuracy | 0.000 (0/7) | 0.714 (5/7) | 0.000 (0/3) | 0.333 (1/3) |
| Displayed claims matching the catalog | 1.000 (1680/1680) | 1.000 (1344/1344) | 1.000 (840/840) | 1.000 (756/756) |
| Recall@10 (application retriever) | 0.455 | 0.739 | 0.398 | 0.578 |
| MRR@3 (application retriever) | 0.450 | 0.967 | 0.875 | 0.938 |
| nDCG@3 (application retriever) | 0.338 | 0.941 | 0.550 | 0.896 |

Before the work, Persian constraints were not read at all, so a request such as "under 900 dollars"
was silently ignored: 22 of the 120 products returned on test
broke a constraint the user had stated, and amounts in toman were not even questioned.

## Persian next to English (test split)

Same intents, ids, constraints and relevance labels in both languages.

| Measure (test split) | Persian | English |
|---|---|---|
| Recall@10 (application retriever) | 0.739 | 0.721 |
| MRR@3 (application retriever) | 0.967 | 0.983 |
| nDCG@3 (application retriever) | 0.941 | 0.921 |
| Constraint violations | 0.000 (0/96) | 0.000 (0/96) |
| Interpretation accuracy | 1.000 (33/33) | 1.000 (33/33) |
| Status accuracy | 0.950 (38/40) | 0.950 (38/40) |
| Clarification accuracy | 0.714 (5/7) | 0.714 (5/7) |
| Abstention on irrelevant queries | 0.000 (0/2) | 0.000 (0/2) |
| Displayed claims matching the catalog | 1.000 (1344/1344) | 1.000 (1344/1344) |

**Do not read this as "Persian works better than English".** The two are close: Persian is slightly
ahead on Recall@10 and nDCG@3 and slightly behind on MRR@3. There are at least two untested
explanations for the differences, neither of which is better retrieval: the Persian lexicon was built directly from the catalog's controlled vocabulary by the
same author who wrote the queries, and unknown Persian words add no features, whereas unknown English
words add character n-gram noise.

### By category (test)

| Category | n | Persian nDCG@3 | English nDCG@3 | Persian MRR@3 | English MRR@3 |
|---|---|---|---|---|---|
| constraints | 6 | 0.967 | 1.000 | 1.000 | 1.000 |
| direct | 6 | 0.985 | 0.985 | 1.000 | 1.000 |
| injection | 3 | 1.000 | 0.844 | 1.000 | 0.833 |
| model-name | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| negation | 2 | 0.765 | 0.667 | 1.000 | 1.000 |
| paraphrase | 6 | 0.833 | 0.849 | 0.833 | 1.000 |
| unknown-spec | 3 | 1.000 | 0.922 | 1.000 | 1.000 |

Counts of 2 to 6 per category are illustrations, not estimates.

### Coverage of the lexicon decides the quality (test)

| Persian test queries | n | builtin MRR@3 | builtin nDCG@3 | builtin Recall@10 |
|---|---|---|---|---|
| use at least one word the Persian lexicon knows | 25 | 1.000 | 0.970 | 0.721 |
| use no word the lexicon knows | 5 | 0.800 | 0.800 | 0.831 |

Quality depends strongly on whether the query contains a word the author's lexicon lists. Coverage
is **measured**, not authored: 25 of 30 positive Persian test queries contain such a word,
which says more about the author's phrasing than about real users.

## Findings

**What works**

- **Constraints are enforced**: 0.000 (0/96) violations on test and
  0.000 (0/54) on dev; every stated constraint was interpreted correctly
  (1.000 (33/33)), including Persian digits, the `٫` and `٬` separators and a slash
  decimal. Exact model names written in Latin letters inside Persian sentences were retrieved correctly
  (nDCG@3 1.000 on 4 queries).
- **Toman is handled by policy**: the toman query produced a question ("Only USD is supported...")
  instead of an ignored or invented budget.
- Every displayed claim matched the catalog (1.000 (1344/1344)).
- Embedded instruction text in Persian did not change eligibility, and ranking quality on those
  queries was not lower than in English (nDCG@3 1.000 against 0.844).

**What does not**

- **A Persian request with no lexicon word retrieves nothing meaningful.** `P4` ("run the latest titles
  at high settings", no word the lexicon knows) scored nDCG@3 0.00: with no features the query vector is
  arbitrary. A real language model would handle this; this approach cannot.
- **Abstention still does not work**: 0.000 (0/2) irrelevant queries were refused (`I3` and `I4`
  were answered), exactly as in English. The cutoff stays off.
- **Negation is not understood**: `G2` ("a programming laptop that is not heavy") scored nDCG@3 0.53.
- **Persian requests cost about 1.4x the time of their English twins** (median of
  46 matched pairs, IQR 1.23-1.54; 1.49 ms against 1.16 ms), from the extra
  pattern matching. Comparing the two evaluation reports' p50 values directly suggests a larger gap,
  but they cover different queries; only matched pairs are comparable.
- **Display**: the UI shows the normalized request topic, in which the zero-width non-joiner became a
  space ("برنامه نویسی"); everything else on screen is English.

## What these numbers do not show

- Quality on real Persian users: the queries are AI-written, unreviewed, and share an author with the lexicon.
- Transliterated Persian ("laptop baraye bazi"), colloquial or misspelled Persian, number words
  ("نهصد دلار"), other Persian dialects, Dari or Arabic.
- Persian responses: none exist yet, so a Persian speaker gets English explanations.
- Statistical significance: 30 positive test queries, no confidence intervals.

## Asking a native speaker to review

`docs/fa_review_sheet.md` lists all 60 queries with their English parallel and the
constraints a correct reading must extract, with columns for a native speaker to mark naturalness
and meaning. Until that review is done, treat this track as a **prototype**.

## Reproduce

```bash
python scripts/build_eval_set_fa.py
python -m recommender.cli evaluate --split dev  --queries data/eval/fa/queries.jsonl --meta data/eval/fa/meta.json --runs 20 --output reports/fa/eval-dev.json
python -m recommender.cli evaluate --split test --queries data/eval/fa/queries.jsonl --meta data/eval/fa/meta.json --runs 20 --output reports/fa/eval-test.json
python scripts/measure_latency_pairs.py
python scripts/render_evaluation_fa_md.py
```

The "before" reports were produced at commit `c2d5ee0` (the code at that commit has no Persian
support); they are kept as published and are not regenerated.
