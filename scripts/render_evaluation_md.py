"""Render docs/evaluation.md from the raw reports in reports/. Numbers are never typed by hand.

Qualitative statements are asserted against the data before they are written, so a regenerated
document cannot silently claim something the new numbers do not support.

Usage: python scripts/render_evaluation_md.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "docs" / "evaluation.md"

dev = json.loads((REPORTS / "eval-dev.json").read_text())
test = json.loads((REPORTS / "eval-test.json").read_text())
ref = json.loads((REPORTS / "eval-test-no-cutoff.json").read_text())

METHODS = [("builtin", "Application retriever (built-in embedder)"), ("keyword", "Keyword baseline (BM25)"),
           ("post_filter", "Ablation: global top-10 filtered afterwards"), ("random", "Random order (floor)")]  # fmt: skip


def m(metric: dict, digits: int = 3) -> str:
    return "n/a" if metric["mean"] is None else f"{metric['mean']:.{digits}f}"


def r(rate: dict) -> str:
    if rate["value"] is None:
        return "n/a (0/0)"
    return f"{rate['value']:.3f} ({rate['numerator']}/{rate['denominator']})"


def retrieval_table(report: dict) -> str:
    s = report["summary"]["retrieval"]
    n = s["builtin"]["recall_at_10"]["n"]
    rows = ["| Method | Recall@10 | MRR@3 | nDCG@3 |", "|---|---|---|---|"]
    for key, label in METHODS:
        x = s[key]
        rows.append(
            f"| {label} | {m(x['recall_at_10'])} | {m(x['mrr_at_3'])} | {m(x['ndcg_at_3'])} |"
        )
    rows.append(f"| Oracle ceiling for Recall@10 | {m(s['oracle_recall_at_10'])} | - | - |")
    return "\n".join(rows) + f"\n\n*n = {n} queries with at least one relevant eligible product.*"


def category_table(report: dict) -> str:
    rows = [
        "| Category | n | builtin R@10 | builtin MRR@3 | builtin nDCG@3 | keyword R@10 | keyword MRR@3 | keyword nDCG@3 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cat, v in report["summary"]["retrieval_by_category"].items():
        b, k = v["builtin"], v["keyword"]
        rows.append(
            f"| {cat} | {v['queries']} | {m(b['recall_at_10'])} | {m(b['mrr_at_3'])} | {m(b['ndcg_at_3'])} | {m(k['recall_at_10'])} | {m(k['mrr_at_3'])} | {m(k['ndcg_at_3'])} |"
        )
    return "\n".join(rows)


def lexicon_table(report: dict) -> str:
    rows = [
        "| Query set | n | builtin MRR@3 | builtin nDCG@3 | keyword MRR@3 | keyword nDCG@3 |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "lexicon_hit_expected": "uses words the concept lexicon knows",
        "lexicon_miss_expected": "uses no word the lexicon knows",
    }
    for key, v in report["summary"]["retrieval_by_lexicon"].items():
        b, k = v["builtin"], v["keyword"]
        rows.append(
            f"| {labels[key]} | {v['queries']} | {m(b['mrr_at_3'])} | {m(b['ndcg_at_3'])} | {m(k['mrr_at_3'])} | {m(k['ndcg_at_3'])} |"
        )
    return "\n".join(rows)


SYSTEM_KEYS = [
    ("status_accuracy", "Status accuracy (ok / no_match / needs_clarification)"),
    ("interpretation_accuracy", "Constraint interpretation accuracy"),
    (
        "constraint_violation_rate",
        "Constraint violation rate (returned products failing a hard constraint)",
    ),
    ("answer_rate_on_positive_queries", "Answer rate on answerable queries"),
    ("false_empty_rate", "False-empty rate (no_match on an answerable query)"),
    ("no_match_accuracy", "No-match accuracy"),
    ("clarification_accuracy", "Clarification accuracy"),
    ("abstention_on_irrelevant", "Abstention on irrelevant queries"),
    ("evidence_validity", "Evidence validity (displayed claims audited against the catalog)"),
]


def system_table() -> str:
    cols = [("Dev (no cutoff)", dev), ("Test (cutoff)", test), ("Test (no cutoff)", ref)]
    rows = ["| Metric | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for key, label in SYSTEM_KEYS:
        rows.append(
            f"| {label} | " + " | ".join(r(rep["summary"]["system"][key]) for _, rep in cols) + " |"
        )
    for key, label in (
        ("mrr_at_3", "MRR@3 of the returned top 3"),
        ("ndcg_at_3", "nDCG@3 of the returned top 3"),
    ):
        rows.append(
            f"| {label} | "
            + " | ".join(
                f"{m(rep['summary']['system'][key])} (n={rep['summary']['system'][key]['n']})"
                for _, rep in cols
            )
            + " |"
        )
    return "\n".join(rows)


def per_query(report: dict) -> dict:
    return {q["query_id"]: q for q in report["per_query"]}


# --- facts the narrative relies on, asserted against the data ---------------------------------------
tr, dr = test["summary"]["retrieval"], dev["summary"]["retrieval"]
assert tr["builtin"]["recall_at_10"]["mean"] > tr["post_filter"]["recall_at_10"]["mean"], (
    "filter-first claim"
)
assert tr["builtin"]["recall_at_10"]["mean"] > tr["keyword"]["recall_at_10"]["mean"], (
    "builtin>keyword on test"
)
assert dr["keyword"]["mrr_at_3"]["mean"] > dr["builtin"]["mrr_at_3"]["mean"], (
    "keyword>builtin MRR on dev"
)
assert test["summary"]["system"]["constraint_violation_rate"]["numerator"] == 0
assert ref["summary"]["system"]["constraint_violation_rate"]["numerator"] == 0
assert test["summary"]["system"]["evidence_validity"]["value"] == 1.0
assert (
    test["summary"]["system"]["abstention_on_irrelevant"]["numerator"]
    < test["summary"]["system"]["abstention_on_irrelevant"]["denominator"]
)
cut, nocut = per_query(test), per_query(ref)
leak = [q for q in test["per_query"] if q["category"] == "irrelevant" and q["status"] == "ok"]
assert leak, "an irrelevant query still answered with the cutoff"
worst_positive = min(
    (q for q in test["per_query"] if q["expected_status"] == "ok" and q["relevant_eligible"] > 0),
    key=lambda q: q["top_score"],
)
assert leak[0]["top_score"] > worst_positive["top_score"], (
    "irrelevant scored above an answerable query"
)
truncated = [
    q["query_id"]
    for q in test["per_query"]
    if q["expected_status"] == "ok" and len(q["returned"]) < len(nocut[q["query_id"]]["returned"])
]
assert truncated, "cutoff truncated some answers"
inj_b = test["summary"]["retrieval_by_category"]["injection"]["builtin"]["ndcg_at_3"]["mean"]
inj_k = test["summary"]["retrieval_by_category"]["injection"]["keyword"]["ndcg_at_3"]["mean"]
assert inj_k > inj_b, "keyword better on injection text"
cal = dev["calibration"]

# Negation: one clear failure (G2) and one query where the two methods disagree (G3).
g2 = per_query(test)["G2"]["retrieval"]
g3 = per_query(test)["G3"]["retrieval"]
assert g3["builtin"]["ndcg_at_3"] > g3["keyword"]["ndcg_at_3"], "G3: builtin ahead of keyword"
assert g2["builtin"]["ndcg_at_3"] < 0.5 and g2["keyword"]["ndcg_at_3"] < 0.5, (
    "G2 is a failure for both methods"
)
neg = test["summary"]["retrieval_by_category"]["negation"]
# Lexicon hit/miss and the 'ahead on all three metrics' claim.
lex = test["summary"]["retrieval_by_lexicon"]
assert (
    lex["lexicon_miss_expected"]["builtin"]["ndcg_at_3"]["mean"]
    < lex["lexicon_hit_expected"]["builtin"]["ndcg_at_3"]["mean"]
)
assert (
    lex["lexicon_miss_expected"]["keyword"]["ndcg_at_3"]["mean"]
    > lex["lexicon_miss_expected"]["builtin"]["ndcg_at_3"]["mean"]
)
for metric in ("recall_at_10", "mrr_at_3", "ndcg_at_3"):
    assert tr["builtin"][metric]["mean"] > tr["keyword"][metric]["mean"], metric
# Ambiguous and contradictory requests always produced a question.
asked = [
    q
    for rep in (dev, test)
    for q in rep["per_query"]
    if q["category"] in ("ambiguous", "contradictory")
]
assert asked and all(q["status"] == "needs_clarification" for q in asked), "clarification claim"
op, cfg = test["operational"], test["configuration"]

DOC = f"""# Evaluation

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
| Catalog | `data/demo/products.jsonl`, {cfg["catalog_size"]} synthetic products, sha256 `{cfg["catalog_sha256"][:12]}` |
| Evaluation set | `data/eval/queries.jsonl`, 60 queries: 20 development + 40 held-out test, sha256 `{cfg["eval_set_sha256"][:12]}` |
| Test labels | frozen; digest `{cfg["test_labels_sha256"][:12]}` is verified on every run |
| Embedder | `{cfg["embedder"]}` (hashed TF-IDF + curated concept lexicon; **not** neural), revision `{cfg["embedder_revision"]}` |
| Explanations | deterministic templates; no language model was used or evaluated |
| Retrieval | exact search over the eligible subset, `retrieval_k` = {cfg["retrieval_k"]}, up to 3 shown |
| Hardware | {op["hardware"]["platform"]}, {op["hardware"]["cpu_count"]} CPUs, Python {op["hardware"]["python"]}, faiss {op["hardware"]["faiss"]} |

Relevance grades are 2 (strong), 1 (partial), 0 (irrelevant), assigned by rules over catalog tags
and specifications that were written independently of the retriever. Recall@10 counts relevant
**eligible** products found in the top 10; MRR@3 and nDCG@3 use graded relevance (gain 2^grade-1,
discount log2(rank+1)). Queries with no relevant eligible product are excluded from the ranking
averages and counted separately.

## Order of events (no tuning on the test split)

1. The abstention cutoff was calibrated on the **development** split only
   (`evaluate --calibrate`): threshold **{cal["threshold"]:.4f}**, balanced accuracy
   {cal["balanced_accuracy"]:.2f}, separable on dev: {cal["separable"]}
   (best irrelevant score {cal["max_abstain_score"]:.3f}, lowest answerable score
   {cal["min_answer_score"]:.3f}). Only {cal["abstain_queries"]} irrelevant development queries
   existed, so this calibration is statistically weak.
2. The **test** split was then run once with that cutoff, and once without it as a reference. Both
   configurations were chosen before looking at test results; nothing was changed afterwards.
3. The cutoff did not generalize well enough to be enabled (see "Findings"), so
   `MIN_SIMILARITY` stays **unset by default**. This decision was made after seeing the test
   result, and the calibration command remains available.

## Retrieval quality (held-out test split)

Retrieval metrics are computed on the ground-truth eligible products so they isolate ranking from
constraint parsing, and they do not depend on the cutoff.

{retrieval_table(test)}

Development split, for reference:

{retrieval_table(dev)}

How to read this:

- Filtering **before** searching beats searching globally and filtering afterwards (Recall@10
  {m(tr["builtin"]["recall_at_10"])} vs {m(tr["post_filter"]["recall_at_10"])} on test), which is why
  the design filters first.
- On the test split the built-in embedder is ahead of the BM25 baseline on all three metrics, but
  on the development split BM25 has the higher MRR@3 ({m(dr["keyword"]["mrr_at_3"])} vs
  {m(dr["builtin"]["mrr_at_3"])}). With 16 and 30 queries the two methods should be considered
  **comparable**, not ranked.
- Recall@10 is capped by the number of relevant products: where more than 10 products are relevant
  the ceiling is below 1. Compare Recall@10 with the oracle ceiling row, not with 1.0.

### By query category (test)

{category_table(test)}

Per-category counts are tiny (2 to 6); treat them as illustrations, not estimates.

### Does the lexicon help? (test)

{lexicon_table(test)}

The expected-lexicon flag is the author's own judgment of whether a query uses a word the curated
lexicon knows. Queries outside it are the fairer test of generalization, and there the built-in
embedder has no clear advantage over keyword search.

## End-to-end system

Through the full service (parsing, filtering, retrieval, evidence, grounding, rendering).

{system_table()}

## Findings

**What holds up**

- **Hard constraints were never violated**: 0 of {test["summary"]["system"]["constraint_violation_rate"]["denominator"]}
  returned products on test (0 of {dev["summary"]["system"]["constraint_violation_rate"]["denominator"]} on dev), including queries with
  unknown required specifications and with instructions embedded in the query text.
- **Every displayed claim was valid**: {test["summary"]["system"]["evidence_validity"]["numerator"]} of
  {test["summary"]["system"]["evidence_validity"]["denominator"]} claims on test matched the catalog under an audit that is independent of the
  grounding validator. This measures grounding against the catalog, not that the catalog is right
  or that a reason is *useful*.
- Constraints written in the query text or passed as controls were interpreted correctly on every
  evaluated query ({r(test["summary"]["system"]["interpretation_accuracy"])}); ambiguous,
  contradictory and unsupported requests produced a question rather than a guess.

**What does not**

- **Abstention on irrelevant queries is unreliable.** With the calibrated cutoff
  {cal["threshold"]:.4f}, {r(test["summary"]["system"]["abstention_on_irrelevant"])} irrelevant test
  queries abstained; without a cutoff, {r(ref["summary"]["system"]["abstention_on_irrelevant"])}.
  `{leak[0]["query_id"]}` ("{leak[0]["query"]}") scored {leak[0]["top_score"]:.3f}, **above** the
  lowest-scoring answerable query ({worst_positive["query_id"]}: {worst_positive["top_score"]:.3f}), so no raw-score cutoff separates
  them on this embedder.
- **The cutoff has a cost.** It cut the results of {len(truncated)} answerable queries
  ({", ".join(truncated)}), both paraphrases that use no word the lexicon knows; the returned
  nDCG@3 fell from {m(ref["summary"]["system"]["ndcg_at_3"])} to {m(test["summary"]["system"]["ndcg_at_3"])}.
- **Negation is not handled.** "coding laptop that is not heavy" scored nDCG@3
  {g2["builtin"]["ndcg_at_3"]:.2f} with the built-in embedder and {g2["keyword"]["ndcg_at_3"]:.2f} with BM25. On "office laptop
  without a dedicated graphics card" the built-in embedder scored {g3["builtin"]["ndcg_at_3"]:.2f} while BM25 scored
  {g3["keyword"]["ndcg_at_3"]:.2f}: keyword matching is pulled toward products that mention dedicated graphics, and the
  built-in embedder was not, most likely because "office" outweighed the rest. Neither method
  understands "without"; with two queries these are anecdotes, and the category averages
  ({m(neg["builtin"]["ndcg_at_3"])} vs {m(neg["keyword"]["ndcg_at_3"])}) should not be read as a measure of negation ability.
- **Instruction-like text lowers ranking quality.** On the three queries with embedded
  instructions, nDCG@3 was {inj_b:.3f} for the built-in embedder against {inj_k:.3f} for BM25. The
  likely cause is that the extra words add character n-gram features to the query; that
  explanation was not isolated or tested. This is a quality problem, not a safety one: no
  ineligible product was returned.
- **Queries that use no word the concept lexicon knows rank worse for the built-in embedder**
  (nDCG@3 {m(lex["lexicon_miss_expected"]["builtin"]["ndcg_at_3"])} against
  {m(lex["lexicon_hit_expected"]["builtin"]["ndcg_at_3"])} for queries that do), and on that subset
  BM25 is slightly ahead ({m(lex["lexicon_miss_expected"]["keyword"]["ndcg_at_3"])}).

## Operational measurements (test run)

| Measure | Value |
|---|---|
| Warm end-to-end latency | p50 {op["warm_p50_ms"]:.2f} ms, p95 {op["warm_p95_ms"]:.2f} ms ({op["warm_request_count"]} warm requests) |
| Cold first request | {op["cold_first_request_ms"]:.2f} ms |
| Slowest stage (mean) | `{op["slowest_stage"]}` ({op["stage_mean_ms"][op["slowest_stage"]]:.3f} ms); all stage means: {", ".join(f"{k} {v:.3f}" for k, v in op["stage_mean_ms"].items())} ms |
| Catalog build (60 products) | {op["catalog_build_seconds"]:.3f} s |
| Peak memory | {op["peak_rss_mb"]:.0f} MB resident (whole process) |
| Requests issued | {op["request_count"]} |
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
python -m recommender.cli evaluate --split test --min-score {cal["threshold"]:.4f} --runs 20 --output reports/eval-test.json
python -m recommender.cli evaluate --split test --runs 20 --output reports/eval-test-no-cutoff.json
python scripts/render_evaluation_md.py
```

The evaluation reads neither `.env` nor environment variables, refuses to run if the catalog or the
frozen test labels changed, and writes per-query rankings and outputs next to the aggregates in
each report.
"""

OUT.write_text(DOC, encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)} ({len(DOC.splitlines())} lines)")
