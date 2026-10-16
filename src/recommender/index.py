"""Persist and load the artifact set: FAISS index, vectors, ID mapping, catalog, manifest.

A build is written into a fresh temporary directory, fully verified, and only then
activated as one version. Loading refuses mismatched or incomplete artifact sets.
Only trusted local files are loaded: JSON and NumPy arrays with allow_pickle=False.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import faiss
import numpy as np

from recommender.catalog import Catalog, parse_catalog
from recommender.documents import DOCUMENT_FORMAT_VERSION, catalog_documents
from recommender.embeddings import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    Embedder,
    EmbeddingError,
    validate_vectors,
)

SCHEMA_VERSION = 1
INDEX_FILE = "products.faiss"
VECTORS_FILE = "embeddings.npy"
IDS_FILE = "product_ids.json"
CATALOG_FILE = "catalog.jsonl"
MANIFEST_FILE = "manifest.json"
CHECKSUMMED_FILES = (INDEX_FILE, VECTORS_FILE, IDS_FILE, CATALOG_FILE)


class ArtifactError(Exception):
    """Raised when an artifact set is missing, incomplete, mismatched or corrupt."""


@dataclass
class Artifacts:
    catalog: Catalog
    vectors: np.ndarray
    index: faiss.Index
    manifest: dict

    @property
    def version(self) -> str:
        return self.manifest["catalog_sha256"][:12]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "faiss-cpu", "pydantic", "sentence-transformers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_artifacts(catalog: Catalog, embedder: Embedder, target_dir: str | Path) -> Path:
    """Embed the catalog and write a verified artifact set into target_dir."""
    target = Path(target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        vectors = embedder.encode_products(catalog_documents(catalog))
        validate_vectors(vectors, expected_rows=len(catalog))

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(tmp / INDEX_FILE))
        np.save(tmp / VECTORS_FILE, vectors, allow_pickle=False)
        (tmp / IDS_FILE).write_text(json.dumps(catalog.product_ids), encoding="utf-8")
        (tmp / CATALOG_FILE).write_bytes(catalog.canonical_bytes())

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "catalog_sha256": catalog.sha256,
            "row_count": len(catalog),
            "vector_dim": int(vectors.shape[1]),
            "embedding_model": embedder.model_id,
            "embedding_revision": embedder.revision,
            "passage_prefix": PASSAGE_PREFIX,
            "query_prefix": QUERY_PREFIX,
            "normalized": True,
            "document_format_version": DOCUMENT_FORMAT_VERSION,
            "built_at": datetime.now(UTC).isoformat(),
            "dependencies": _dependency_versions(),
            "files": {name: _sha256_file(tmp / name) for name in CHECKSUMMED_FILES},
        }
        (tmp / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        load_artifacts(tmp)  # verify the complete set before activating it

        previous = None
        if target.exists():
            previous = target.parent / f".{target.name}.old-{uuid.uuid4().hex[:8]}"
            target.rename(previous)
        tmp.rename(target)
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def load_artifacts(directory: str | Path) -> Artifacts:
    """Load and verify an artifact set. Raises ArtifactError with a rebuild instruction."""
    root = Path(directory)
    rebuild = "rebuild the artifacts with the build-index command"

    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ArtifactError(f"{MANIFEST_FILE} missing in {root}; {rebuild}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{MANIFEST_FILE} is not valid JSON; {rebuild}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported manifest schema version; {rebuild}")

    for name in CHECKSUMMED_FILES:
        path = root / name
        if not path.is_file():
            raise ArtifactError(f"{name} missing in {root}; {rebuild}")
        if _sha256_file(path) != manifest.get("files", {}).get(name):
            raise ArtifactError(f"checksum mismatch for {name}; {rebuild}")

    try:
        catalog = parse_catalog((root / CATALOG_FILE).read_text(encoding="utf-8"))
        product_ids = json.loads((root / IDS_FILE).read_text(encoding="utf-8"))
        vectors = np.load(root / VECTORS_FILE, allow_pickle=False)
        index = faiss.read_index(str(root / INDEX_FILE))
        validate_vectors(
            vectors, expected_rows=manifest["row_count"], expected_dim=manifest["vector_dim"]
        )
    except (ValueError, OSError, RuntimeError, KeyError, EmbeddingError) as exc:
        raise ArtifactError(f"artifact set is unreadable or invalid ({exc}); {rebuild}") from exc

    if catalog.sha256 != manifest["catalog_sha256"]:
        raise ArtifactError(f"catalog hash does not match the manifest; {rebuild}")
    if product_ids != catalog.product_ids:
        raise ArtifactError(f"product ID mapping does not match the catalog order; {rebuild}")
    if len(catalog) != manifest["row_count"] or index.ntotal != manifest["row_count"]:
        raise ArtifactError(f"row counts disagree across artifacts; {rebuild}")
    if index.d != manifest["vector_dim"]:
        raise ArtifactError(f"index dimension does not match the manifest; {rebuild}")
    if not np.allclose(index.reconstruct_n(0, index.ntotal), vectors, atol=1e-6):
        raise ArtifactError(f"FAISS index does not match the stored vectors; {rebuild}")

    return Artifacts(catalog=catalog, vectors=vectors, index=index, manifest=manifest)


def ensure_embedder_matches(artifacts: Artifacts, embedder: Embedder) -> None:
    """Queries must use the same model and revision that built the vectors."""
    manifest = artifacts.manifest
    if (
        manifest["embedding_model"] != embedder.model_id
        or manifest["embedding_revision"] != embedder.revision
    ):
        raise ArtifactError(
            "embedder differs from the one that built the artifacts "
            f"({manifest['embedding_model']}@{manifest['embedding_revision']}); "
            "rebuild the artifacts"
        )
