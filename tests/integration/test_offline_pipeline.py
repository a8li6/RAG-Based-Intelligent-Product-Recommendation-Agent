"""The default pipeline end to end with the real built-in embedder: no mocks, no network."""

import json
from pathlib import Path

import numpy as np
import pytest

from recommender.catalog import load_catalog
from recommender.cli import main
from recommender.documents import catalog_documents
from recommender.index import (
    MANIFEST_FILE,
    ArtifactError,
    build_artifacts,
    load_artifacts,
)
from recommender.local_embedder import STATE_JSON, STATE_NPY, LocalEmbedder
from recommender.retrieval import Retriever
from recommender.schemas import Constraints

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"
REFERENCE = (
    "I need a laptop for Python development and travel, under $900, with at least 16 GB RAM "
    "and weight at most 1.5 kg."
)


@pytest.fixture
def built(tmp_path):
    return build_artifacts(load_catalog(FIXTURE), LocalEmbedder(), tmp_path / "catalog-v1")


def test_build_stores_the_fitted_state_with_checksums(built):
    manifest = json.loads((built / MANIFEST_FILE).read_text())
    assert manifest["embedding_model"] == "builtin-hashed-tfidf-v1"
    assert manifest["embedder_state_files"] == sorted([STATE_JSON, STATE_NPY])
    assert STATE_JSON in manifest["files"] and STATE_NPY in manifest["files"]
    assert manifest["revision_pinned"] is True
    assert manifest["embedding_revision"].startswith("idf-")


def test_a_fresh_embedder_restores_state_from_the_artifacts(built):
    artifacts = load_artifacts(built)
    fresh = LocalEmbedder()
    assert fresh.revision == "unfitted"
    retriever = Retriever(artifacts, fresh)
    assert fresh.revision == artifacts.manifest["embedding_revision"]
    rows = retriever.eligible_rows(Constraints())
    candidates = retriever.search("portable laptop for coding and commuting", rows)
    assert candidates[0].product_id in {"P004", "P001"}
    stored = artifacts.vectors
    np.testing.assert_allclose(
        fresh.encode_products(catalog_documents(artifacts.catalog)),
        stored,
        atol=1e-6,
    )


def test_tampered_state_is_detected(built):
    (built / STATE_NPY).write_bytes((built / STATE_NPY).read_bytes()[:-8] + b"\x00" * 8)
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifacts(built)


def test_missing_state_file_is_detected(built):
    (built / STATE_JSON).unlink()
    with pytest.raises(ArtifactError, match="missing"):
        load_artifacts(built)


def test_state_from_a_different_build_is_rejected(tmp_path, built):
    other = LocalEmbedder().fit(["only one document about gaming"])
    artifacts = load_artifacts(built)
    artifacts.embedder_state = other.state_files()
    with pytest.raises(ArtifactError, match="rebuild"):
        Retriever(artifacts, LocalEmbedder())


def test_default_cli_builds_and_recommends_with_no_injection(tmp_path, capsys):
    out = tmp_path / "catalog-v1"
    assert main(["build-index", "--input", str(FIXTURE), "--output", str(out)]) == 0
    captured = capsys.readouterr()
    assert "built 8 products" in captured.out
    assert "not reproducible" not in captured.err  # builtin revision is a content hash

    assert main(["recommend", "--query", REFERENCE, "--artifacts", str(out)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "ok"
    assert body["explanation_mode"] == "deterministic"
    assert {r["product_id"] for r in body["recommendations"]} == {"P001", "P004"}
    assert all(r["reasons"] for r in body["recommendations"])


def test_default_cli_search_shows_scores(tmp_path, capsys):
    out = tmp_path / "catalog-v1"
    assert main(["build-index", "--input", str(FIXTURE), "--output", str(out)]) == 0
    capsys.readouterr()
    assert main(["search", "--query", "laptop for gaming", "--artifacts", str(out),
                 "--include-out-of-stock"]) == 0  # fmt: skip
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("eligible: 8 of 8")
    assert lines[1].split()[0] in {"P002", "P005"}


def test_rebuild_is_reproducible(tmp_path):
    first = build_artifacts(load_catalog(FIXTURE), LocalEmbedder(), tmp_path / "a")
    second = build_artifacts(load_catalog(FIXTURE), LocalEmbedder(), tmp_path / "b")
    m1 = json.loads((first / MANIFEST_FILE).read_text())
    m2 = json.loads((second / MANIFEST_FILE).read_text())
    for key in ("catalog_sha256", "embedding_revision", "files"):
        assert m1[key] == m2[key]
    np.testing.assert_array_equal(
        np.load(first / "embeddings.npy"), np.load(second / "embeddings.npy")
    )
