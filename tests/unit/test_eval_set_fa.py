import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from recommender.catalog import load_catalog
from recommender.schemas import Constraints

ROOT = Path(__file__).resolve().parents[2]
FA = ROOT / "data" / "eval" / "fa"
PERSIAN_LETTER = re.compile(r"[؀-ۿ]")
PERSIAN_DIGIT = re.compile(r"[۰-۹]")


@pytest.fixture(scope="module")
def rows():
    return [
        json.loads(x)
        for x in (FA / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


@pytest.fixture(scope="module")
def english():
    lines = (ROOT / "data" / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(x)["query_id"]: json.loads(x) for x in lines if x.strip()}


@pytest.fixture(scope="module")
def meta():
    return json.loads((FA / "meta.json").read_text(encoding="utf-8"))


def test_committed_files_match_the_generator():
    spec = importlib.util.spec_from_file_location(
        "fa_gen", ROOT / "scripts" / "build_eval_set_fa.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    queries, meta_text = gen.render(*gen.build())
    assert (FA / "queries.jsonl").read_text(encoding="utf-8") == queries
    assert (FA / "meta.json").read_text(encoding="utf-8") == meta_text


def test_sizes_splits_and_catalog_hash(rows, meta):
    assert len(rows) == 60 == meta["query_count"]
    assert Counter(r["split"] for r in rows) == {"dev": 20, "test": 40}
    assert meta["catalog_sha256"] == load_catalog(ROOT / "data" / "demo" / "products.jsonl").sha256


def test_every_query_is_parallel_to_an_english_query(rows, english):
    assert {r["query_id"] for r in rows} == set(english)
    for r in rows:
        en = english[r["query_id"]]
        assert r["split"] == en["split"] and r["group"] == en["group"], r["query_id"]
        assert r["category"] == en["category"], r["query_id"]
        assert r["constraints"] == en["constraints"] and r["controls"] == en["controls"], r[
            "query_id"
        ]
        assert r["relevance"] == en["relevance"] and r["eligible_ids"] == en["eligible_ids"], r[
            "query_id"
        ]
        assert r["expected_status"] == en["expected_status"], r["query_id"]


def test_every_query_is_actually_persian_and_texts_are_unique(rows):
    for r in rows:
        assert PERSIAN_LETTER.search(r["query"]), r["query_id"]
    assert len({r["query"] for r in rows}) == 60
    english_texts = {
        json.loads(x)["query"]
        for x in (ROOT / "data" / "eval" / "queries.jsonl").read_text().splitlines()
    }
    assert not ({r["query"] for r in rows} & english_texts)


def test_the_set_exercises_persian_specific_phrasing(rows):
    text = "\n".join(r["query"] for r in rows)
    assert len([r for r in rows if PERSIAN_DIGIT.search(r["query"])]) >= 12  # Persian digits
    assert "٫" in text or "۱٫" in text  # Persian decimal separator
    assert "٬" in text  # Persian thousands separator
    assert re.search(r"[۰-۹]/[۰-۹]", text)  # slash used as a decimal
    assert "تومان" in text  # currency policy case
    assert "‌" in text  # zero-width non-joiner
    assert "لپ تاپ" in text and "لپتاپ" in text  # spelling variants of "laptop"
    assert "گیگ" in text and "گیگابایت" in text and "کیلو " in text and "کیلوگرم" in text


def test_the_toman_query_must_ask_not_guess(rows):
    a4 = next(r for r in rows if r["query_id"] == "A4")
    assert "تومان" in a4["query"] and a4["expected_status"] == "needs_clarification"
    assert "Persian-specific" in a4["notes"]


def test_test_labels_are_frozen_by_digest(rows, meta):
    frozen = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["query_id"])
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    assert digest == meta["test_labels_sha256"]


def test_eligibility_is_recomputed_from_the_stored_constraints(rows):
    catalog = load_catalog(ROOT / "data" / "demo" / "products.jsonl")
    for r in rows:
        expected = [
            catalog.products[i].product_id
            for i in catalog.eligible_rows(Constraints(**r["constraints"]))
        ]
        assert r["eligible_ids"] == expected, r["query_id"]


def test_the_set_discloses_its_authorship_and_review_status(meta):
    assert meta["reviewed_by_native_speaker"] is False
    assert "AI assistant" in meta["authorship"] and "NOT been reviewed" in meta["authorship"]
    assert "responses remain English" in meta["scope"]
    assert "Single author" in meta["labeling"]


def test_lexicon_coverage_is_computed_later_not_authored(rows):
    assert all(r["lexicon_hit_expected"] is None for r in rows)
