"""Render docs/evaluation_fa.md and docs/fa_review_sheet.md from the raw reports and data.

Numbers are never typed by hand, and qualitative statements are asserted against the data before
they are written, so a regenerated document cannot claim something the numbers do not support.

Usage: python scripts/render_evaluation_fa_md.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
FA_DIR = ROOT / "data" / "eval" / "fa"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


base_dev, base_test = (
    load(R / "fa" / "baseline-no-persian-support-dev.json"),
    load(R / "fa" / "baseline-no-persian-support-test.json"),
)
fa_dev, fa_test = load(R / "fa" / "eval-dev.json"), load(R / "fa" / "eval-test.json")
en_test, en_dev = load(R / "eval-test-no-cutoff.json"), load(R / "eval-dev.json")
lat = load(R / "fa" / "latency-pairs.json")
meta = load(FA_DIR / "meta.json")


def m(x: dict, d: int = 3) -> str:
    return "n/a" if x["mean"] is None else f"{x['mean']:.{d}f}"


def r(x: dict) -> str:
    return (
        "n/a (0/0)"
        if x["value"] is None
        else f"{x['value']:.3f} ({x['numerator']}/{x['denominator']})"
    )


def sysm(rep: dict) -> dict:
    return rep["summary"]["system"]


def ret(rep: dict, name: str = "builtin") -> dict:
    return rep["summary"]["retrieval"][name]


def per(rep: dict) -> dict:
    return {q["query_id"]: q for q in rep["per_query"]}


# --- claims asserted before they are written -----------------------------------------------------------------
assert (
    sysm(base_test)["constraint_violation_rate"]["numerator"] > 0
    and sysm(base_dev)["constraint_violation_rate"]["numerator"] > 0
)
for rep in (fa_dev, fa_test):
    assert sysm(rep)["constraint_violation_rate"]["numerator"] == 0
    assert sysm(rep)["interpretation_accuracy"]["value"] == 1.0
    assert sysm(rep)["evidence_validity"]["value"] == 1.0
assert ret(fa_test)["recall_at_10"]["mean"] > ret(base_test)["recall_at_10"]["mean"]
assert abs(ret(fa_test)["recall_at_10"]["mean"] - ret(en_test)["recall_at_10"]["mean"]) < 0.05, (
    "FA and EN recall are close"
)
assert ret(fa_test)["ndcg_at_3"]["mean"] >= ret(en_test)["ndcg_at_3"]["mean"], (
    "FA nDCG is not below EN"
)
fa_lex = fa_test["summary"]["retrieval_by_lexicon"]
assert (
    fa_lex["lexicon_hit_expected"]["builtin"]["ndcg_at_3"]["mean"]
    > fa_lex["lexicon_miss_expected"]["builtin"]["ndcg_at_3"]["mean"]
)
p4 = per(fa_test)["P4"]["retrieval"]["builtin"]["ndcg_at_3"]
assert p4 == 0.0, "P4 (no lexicon word) scores zero"
mismatch = sorted(
    q["query_id"] for q in fa_test["per_query"] if q["status"] != q["expected_status"]
)
assert mismatch == ["I3", "I4"], mismatch
assert sysm(fa_test)["abstention_on_irrelevant"]["numerator"] == 0
inj_fa = fa_test["summary"]["retrieval_by_category"]["injection"]["builtin"]["ndcg_at_3"]["mean"]
inj_en = en_test["summary"]["retrieval_by_category"]["injection"]["builtin"]["ndcg_at_3"]["mean"]
assert inj_fa > inj_en, "Persian injection text does not lower ranking quality"
assert 1.1 < lat["ratio_median"] < 2.0, "matched-pair latency ratio"
fa_t, en_t = ret(fa_test), ret(en_test)
assert (
    fa_t["recall_at_10"]["mean"] > en_t["recall_at_10"]["mean"]
    and fa_t["ndcg_at_3"]["mean"] > en_t["ndcg_at_3"]["mean"]
)
assert fa_t["mrr_at_3"]["mean"] < en_t["mrr_at_3"]["mean"], "Persian MRR@3 is below English"
assert (
    fa_test["summary"]["retrieval_by_category"]["model-name"]["builtin"]["ndcg_at_3"]["mean"] == 1.0
)
assert per(fa_test)["A4"]["status"] == "needs_clarification", "toman must ask"
neg_fa = fa_test["summary"]["retrieval_by_category"]["negation"]["builtin"]["ndcg_at_3"]["mean"]
g2 = per(fa_test)["G2"]["retrieval"]["builtin"]["ndcg_at_3"]
assert g2 < 0.6, "negation query G2 fails"
cfg = fa_test["configuration"]
lex_n = {k: v["queries"] for k, v in fa_test["summary"]["retrieval_by_lexicon"].items()}


def before_after() -> str:
    rows = [
        "| Metric | Test: before | Test: after | Dev: before | Dev: after |",
        "|---|---|---|---|---|",
    ]
    keys = [("constraint_violation_rate", "Constraint violations among returned products"),
            ("interpretation_accuracy", "Constraint interpretation accuracy"),
            ("status_accuracy", "Status accuracy"), ("no_match_accuracy", "No-match accuracy"),
            ("clarification_accuracy", "Clarification accuracy"),
            ("evidence_validity", "Displayed claims matching the catalog")]  # fmt: skip
    for k, label in keys:
        rows.append(
            f"| {label} | {r(sysm(base_test)[k])} | {r(sysm(fa_test)[k])} | {r(sysm(base_dev)[k])} | {r(sysm(fa_dev)[k])} |"
        )
    for k, label in (("recall_at_10", "Recall@10"), ("mrr_at_3", "MRR@3"), ("ndcg_at_3", "nDCG@3")):
        rows.append(
            f"| {label} (application retriever) | {m(ret(base_test)[k])} | {m(ret(fa_test)[k])} | {m(ret(base_dev)[k])} | {m(ret(fa_dev)[k])} |"
        )
    return "\n".join(rows)


def parallel() -> str:
    rows = ["| Measure (test split) | Persian | English |", "|---|---|---|"]
    for k, label in (("recall_at_10", "Recall@10"), ("mrr_at_3", "MRR@3"), ("ndcg_at_3", "nDCG@3")):
        rows.append(
            f"| {label} (application retriever) | {m(ret(fa_test)[k])} | {m(ret(en_test)[k])} |"
        )
    for k, label in (("constraint_violation_rate", "Constraint violations"), ("interpretation_accuracy", "Interpretation accuracy"),
                     ("status_accuracy", "Status accuracy"), ("clarification_accuracy", "Clarification accuracy"),
                     ("abstention_on_irrelevant", "Abstention on irrelevant queries"), ("evidence_validity", "Displayed claims matching the catalog")):  # fmt: skip
        rows.append(f"| {label} | {r(sysm(fa_test)[k])} | {r(sysm(en_test)[k])} |")
    return "\n".join(rows)


def categories() -> str:
    rows = [
        "| Category | n | Persian nDCG@3 | English nDCG@3 | Persian MRR@3 | English MRR@3 |",
        "|---|---|---|---|---|---|",
    ]
    for cat, v in fa_test["summary"]["retrieval_by_category"].items():
        e = en_test["summary"]["retrieval_by_category"][cat]["builtin"]
        rows.append(
            f"| {cat} | {v['queries']} | {m(v['builtin']['ndcg_at_3'])} | {m(e['ndcg_at_3'])} | {m(v['builtin']['mrr_at_3'])} | {m(e['mrr_at_3'])} |"
        )
    return "\n".join(rows)


def lexicon() -> str:
    rows = [
        "| Persian test queries | n | builtin MRR@3 | builtin nDCG@3 | builtin Recall@10 |",
        "|---|---|---|---|---|",
    ]
    for key, label in (("lexicon_hit_expected", "use at least one word the Persian lexicon knows"),
                       ("lexicon_miss_expected", "use no word the lexicon knows")):  # fmt: skip
        v = fa_test["summary"]["retrieval_by_lexicon"][key]
        b = v["builtin"]
        rows.append(
            f"| {label} | {v['queries']} | {m(b['mrr_at_3'])} | {m(b['ndcg_at_3'])} | {m(b['recall_at_10'])} |"
        )
    return "\n".join(rows)


DOC = f"""# Persian (Farsi) evaluation

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
> - {cfg["catalog_size"]}-product synthetic catalog, {meta["splits"]["test"]} held-out test queries, no confidence intervals.

## What was built

| Piece | What it does |
|---|---|
| `persian.py` normalization | Persian and Arabic digits, the Persian decimal (`٫`) and thousands (`٬`) separators, the Persian comma, slash decimals before "kilo", Arabic yeh/kaf, zero-width non-joiner |
| Persian constraint phrases | budgets (زیر، کمتر از، حداکثر، تا، بودجه، «یا کمتر»), RAM minimums, weight limits, in-stock, clearing a limit, and the same clarification questions as English |
| Currency policy | only USD; amounts in **toman**, rial, euro, pound are asked about, never guessed or silently ignored |
| Concept lexicon + translation | Persian words map to the same concept ids as the English lexicon and are translated to the catalog's own use-case vocabulary (every catalog tag has Persian phrases). Unknown Persian words add no retrieval features |

## Order of events

1. The Persian set (60 queries, parallel to the English ones) was written and its test labels frozen
   by digest `{meta["test_labels_sha256"][:12]}` (commit `a34ef91`).
2. **Baseline** with the unchanged system was measured (commit `c2d5ee0`).
3. Persian support was implemented (commits `2fdf851`, `1ee7565`, `9884a59`). The development split was
   run once with the finished support and **no change was made after seeing it**. Changes made while
   implementing came from my own unit and fixture tests, for example the case of Latin `RAM`/`USD`, the
   missing "lighter than" phrase, and a travel query that ranked differently from its English twin
   on the 8-product fixture, not from evaluation results.
4. The **test split was run once**, at commit `51211c0`. Nothing was changed afterwards.

## Before and after Persian support

The baseline is the same system before any Persian support existed.

{before_after()}

Before the work, Persian constraints were not read at all, so a request such as "under 900 dollars"
was silently ignored: {sysm(base_test)["constraint_violation_rate"]["numerator"]} of the {sysm(base_test)["constraint_violation_rate"]["denominator"]} products returned on test
broke a constraint the user had stated, and amounts in toman were not even questioned.

## Persian next to English (test split)

Same intents, ids, constraints and relevance labels in both languages.

{parallel()}

**Do not read this as "Persian works better than English".** The two are close: Persian is slightly
ahead on Recall@10 and nDCG@3 and slightly behind on MRR@3. There are at least two untested
explanations for the differences, neither of which is better retrieval: the Persian lexicon was built directly from the catalog's controlled vocabulary by the
same author who wrote the queries, and unknown Persian words add no features, whereas unknown English
words add character n-gram noise.

### By category (test)

{categories()}

Counts of 2 to 6 per category are illustrations, not estimates.

### Coverage of the lexicon decides the quality (test)

{lexicon()}

Quality depends strongly on whether the query contains a word the author's lexicon lists. Coverage
is **measured**, not authored: {lex_n["lexicon_hit_expected"]} of {lex_n["lexicon_hit_expected"] + lex_n["lexicon_miss_expected"]} positive Persian test queries contain such a word,
which says more about the author's phrasing than about real users.

## Findings

**What works**

- **Constraints are enforced**: {r(sysm(fa_test)["constraint_violation_rate"])} violations on test and
  {r(sysm(fa_dev)["constraint_violation_rate"])} on dev; every stated constraint was interpreted correctly
  ({r(sysm(fa_test)["interpretation_accuracy"])}), including Persian digits, the `٫` and `٬` separators and a slash
  decimal. Exact model names written in Latin letters inside Persian sentences were retrieved correctly
  (nDCG@3 1.000 on {fa_test["summary"]["retrieval_by_category"]["model-name"]["queries"]} queries).
- **Toman is handled by policy**: the toman query produced a question ("Only USD is supported...")
  instead of an ignored or invented budget.
- Every displayed claim matched the catalog ({r(sysm(fa_test)["evidence_validity"])}).
- Embedded instruction text in Persian did not change eligibility, and ranking quality on those
  queries was not lower than in English (nDCG@3 {inj_fa:.3f} against {inj_en:.3f}).

**What does not**

- **A Persian request with no lexicon word retrieves nothing meaningful.** `P4` ("run the latest titles
  at high settings", no word the lexicon knows) scored nDCG@3 {p4:.2f}: with no features the query vector is
  arbitrary. A real language model would handle this; this approach cannot.
- **Abstention still does not work**: {r(sysm(fa_test)["abstention_on_irrelevant"])} irrelevant queries were refused (`I3` and `I4`
  were answered), exactly as in English. The cutoff stays off.
- **Negation is not understood**: `G2` ("a programming laptop that is not heavy") scored nDCG@3 {g2:.2f}.
- **Persian requests cost about {lat["ratio_median"]:.1f}x the time of their English twins** (median of
  {lat["pairs"]} matched pairs, IQR {lat["ratio_q1"]:.2f}-{lat["ratio_q3"]:.2f}; {lat["persian_median_ms"]:.2f} ms against {lat["english_median_ms"]:.2f} ms), from the extra
  pattern matching. Comparing the two evaluation reports' p50 values directly suggests a larger gap,
  but they cover different queries; only matched pairs are comparable.
- **Display**: the UI shows the normalized request topic, in which the zero-width non-joiner became a
  space ("برنامه نویسی"); everything else on screen is English.

## What these numbers do not show

- Quality on real Persian users: the queries are AI-written, unreviewed, and share an author with the lexicon.
- Transliterated Persian ("laptop baraye bazi"), colloquial or misspelled Persian, number words
  ("نهصد دلار"), other Persian dialects, Dari or Arabic.
- Persian responses: none exist yet, so a Persian speaker gets English explanations.
- Statistical significance: {fa_test["summary"]["positive_queries_with_relevant_eligible"]} positive test queries, no confidence intervals.

## Asking a native speaker to review

`docs/fa_review_sheet.md` lists all {meta["query_count"]} queries with their English parallel and the
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
"""
(ROOT / "docs" / "evaluation_fa.md").write_text(DOC, encoding="utf-8")

# --- review sheet ------------------------------------------------------------------------------------------------
fa_rows = [
    json.loads(x)
    for x in (FA_DIR / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    if x.strip()
]
en_rows = {
    q["query_id"]: q
    for q in map(
        json.loads,
        (ROOT / "data" / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines(),
    )
}


def expected(c: dict, controls: dict) -> str:
    parts = []
    if c["max_price_cents"] is not None:
        parts.append(f"max price {c['max_price_cents']} cents")
    if c["min_ram_gb"] is not None:
        parts.append(f"RAM >= {c['min_ram_gb']} GB")
    if c["max_weight_kg"] is not None:
        parts.append(f"weight <= {c['max_weight_kg']:g} kg")
    if controls:
        parts.append("(controls: " + ", ".join(f"{k}={v}" for k, v in controls.items()) + ")")
    return "; ".join(parts) or "none"


sheet = [
    "# Persian query review sheet\n",
    "For a **native Persian speaker**. The queries in `data/eval/fa/queries.jsonl` were written by an AI assistant and have not",
    "been reviewed. For each row please judge: **natural?** (would a real user write this), **meaning?** (does it mean the same as",
    "the English parallel), and give a better wording if not. Do not edit the data files directly: send the corrections and they",
    "will be applied and the test labels re-frozen as a new, versioned set.\n",
    "Expected status: `ok`, `no_match` (no product satisfies the constraints) or `needs_clarification` (must ask).\n",
    "| id | split | Persian query | English parallel | expected status | constraints a correct reading extracts | natural? (y/n) | same meaning? (y/n) | better wording |",
    "|---|---|---|---|---|---|---|---|---|",
]
for q in sorted(fa_rows, key=lambda x: (x["split"], x["query_id"])):
    en = (
        en_rows[q["query_id"]]["query"]
        if q["query_id"] != "A4"
        else "(Persian-specific) laptop under 40 million toman"
    )
    sheet.append(
        f"| {q['query_id']} | {q['split']} | {q['query']} | {en} | {q['expected_status']} | {expected(q['constraints'], q['controls'])} |  |  |  |"
    )
(ROOT / "docs" / "fa_review_sheet.md").write_text("\n".join(sheet) + "\n", encoding="utf-8")
print("wrote docs/evaluation_fa.md and docs/fa_review_sheet.md")
