import json
from pathlib import Path

import numpy as np
import pytest

from recommender.catalog import load_catalog
from recommender.embeddings import EmbeddingError
from recommender.index import (
    CATALOG_FILE,
    IDS_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    VECTORS_FILE,
    ArtifactError,
    build_artifacts,
    ensure_embedder_matches,
    load_artifacts,
)

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"


@pytest.fixture
def catalog():
    return load_catalog(FIXTURE)


@pytest.fixture
def built(tmp_path, catalog, embedder):
    return build_artifacts(catalog, embedder, tmp_path / "catalog-v1")


def test_build_writes_the_full_artifact_set(built):
    for name in (INDEX_FILE, VECTORS_FILE, IDS_FILE, CATALOG_FILE, MANIFEST_FILE):
        assert (built / name).is_file()
    manifest = json.loads((built / MANIFEST_FILE).read_text())
    assert manifest["row_count"] == 8
    assert manifest["embedding_revision"] == "test-rev-1"
    assert manifest["normalized"] is True
    assert set(manifest["files"]) == {INDEX_FILE, VECTORS_FILE, IDS_FILE, CATALOG_FILE}


def test_roundtrip_preserves_ids_and_scores(built, catalog, embedder):
    artifacts = load_artifacts(built)
    assert artifacts.catalog.product_ids == catalog.product_ids
    assert artifacts.catalog.sha256 == catalog.sha256
    query = embedder.encode_query("portable laptop for coding")
    fresh_scores, fresh_pos = artifacts.index.search(query, 8)
    from recommender.documents import catalog_documents

    fresh = embedder.encode_products(catalog_documents(catalog))
    expected = fresh @ query[0]
    np.testing.assert_allclose(np.sort(fresh_scores[0]), np.sort(expected), atol=1e-5)
    assert sorted(fresh_pos[0].tolist()) == list(range(8))


def test_no_temporary_directories_are_left_behind(built):
    leftovers = [p.name for p in built.parent.iterdir() if p.name != built.name]
    assert leftovers == []


def test_rebuild_replaces_previous_version(built, catalog, embedder):
    build_artifacts(catalog, embedder, built)
    assert load_artifacts(built).manifest["row_count"] == 8
    assert [p.name for p in built.parent.iterdir()] == [built.name]


def test_missing_file_is_detected(built):
    (built / VECTORS_FILE).unlink()
    with pytest.raises(ArtifactError, match="missing"):
        load_artifacts(built)


def test_missing_manifest_is_detected(built):
    (built / MANIFEST_FILE).unlink()
    with pytest.raises(ArtifactError, match="manifest"):
        load_artifacts(built)


def test_shuffled_id_mapping_is_detected(built):
    ids = json.loads((built / IDS_FILE).read_text())
    (built / IDS_FILE).write_text(json.dumps(list(reversed(ids))))
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifacts(built)


def test_shuffled_mapping_with_rewritten_checksum_is_still_detected(built):
    ids = json.loads((built / IDS_FILE).read_text())
    (built / IDS_FILE).write_text(json.dumps(list(reversed(ids))))
    manifest = json.loads((built / MANIFEST_FILE).read_text())
    import hashlib

    manifest["files"][IDS_FILE] = hashlib.sha256((built / IDS_FILE).read_bytes()).hexdigest()
    (built / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="mapping"):
        load_artifacts(built)


def test_changed_catalog_is_detected(built):
    lines = (built / CATALOG_FILE).read_text().splitlines()
    record = json.loads(lines[0])
    record["price_cents"] += 100
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    (built / CATALOG_FILE).write_text("\n".join(lines) + "\n")
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifacts(built)


def test_tampered_vectors_are_detected(built):
    vectors = np.load(built / VECTORS_FILE)
    np.save(built / VECTORS_FILE, vectors[::-1].copy())
    with pytest.raises(ArtifactError):
        load_artifacts(built)


def test_vectors_that_disagree_with_index_are_detected(built):
    import hashlib

    vectors = np.load(built / VECTORS_FILE)
    np.save(built / VECTORS_FILE, vectors[::-1].copy())
    manifest = json.loads((built / MANIFEST_FILE).read_text())
    manifest["files"][VECTORS_FILE] = hashlib.sha256(
        (built / VECTORS_FILE).read_bytes()
    ).hexdigest()
    (built / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="index does not match"):
        load_artifacts(built)


def test_corrupt_manifest_json_is_detected(built):
    (built / MANIFEST_FILE).write_text("{nope")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        load_artifacts(built)


def test_embedder_mismatch_is_detected(built, embedder):
    artifacts = load_artifacts(built)
    ensure_embedder_matches(artifacts, embedder)
    embedder.revision = "other"
    with pytest.raises(ArtifactError, match="rebuild"):
        ensure_embedder_matches(artifacts, embedder)


def test_failed_build_leaves_no_partial_directory(tmp_path, catalog):
    class BrokenEmbedder:
        model_id = "x"
        revision = "y"

        def encode_products(self, texts):
            return np.full((len(texts), 4), np.nan, dtype=np.float32)

    with pytest.raises(EmbeddingError):
        build_artifacts(catalog, BrokenEmbedder(), tmp_path / "catalog-v1")
    assert list(tmp_path.iterdir()) == []


def test_manifest_records_build_timing(built):
    manifest = json.loads((built / MANIFEST_FILE).read_text())
    assert manifest["embed_seconds"] >= 0
    assert manifest["build_seconds"] >= manifest["embed_seconds"]


def test_revision_pinning_is_recorded(built, catalog, embedder, tmp_path):
    from recommender.index import is_pinned_revision

    assert json.loads((built / MANIFEST_FILE).read_text())["revision_pinned"] is False
    embedder.revision = "a" * 40
    pinned = build_artifacts(catalog, embedder, tmp_path / "pinned")
    assert json.loads((pinned / MANIFEST_FILE).read_text())["revision_pinned"] is True
    assert is_pinned_revision("0123456789abcdef0123456789abcdef01234567")
    for bad in ("main", "v1.0", "a" * 39, "A" * 40, "g" * 40, ""):
        assert not is_pinned_revision(bad)
