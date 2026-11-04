"""Built-in, fully offline embedder: hashed TF-IDF with a small concept lexicon.

This is NOT a neural model. It maps text to a fixed-size, L2-normalized vector built from
stemmed words, a few hand-written concept tokens (for example "programming" and "coding"
share a concept) and character trigrams (tolerance for spelling variants). Similarity is
therefore lexical plus a small amount of curated vocabulary knowledge. It cannot understand
negation ("not for gaming") or concepts missing from the lexicon.

It needs no download and no service, is deterministic, and its learned state (per-bin IDF
weights fitted on the catalog) is persisted with the index artifacts.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from functools import lru_cache

import numpy as np

from recommender import persian
from recommender.embeddings import EmbeddingError, validate_vectors

MODEL_ID = "builtin-hashed-tfidf-v1"
DIM = 1024
STATE_JSON = "embedder.json"
STATE_NPY = "embedder.npy"

WORD_WEIGHT = 1.0
CONCEPT_WEIGHT = 0.8
TRIGRAM_WEIGHT = 0.12

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "to",
        "of",
        "in",
        "on",
        "at",
        "is",
        "are",
        "be",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "you",
        "need",
        "want",
        "looking",
        "find",
        "get",
        "show",
        "give",
        "please",
        "that",
        "this",
        "which",
        "has",
        "have",
        "under",
        "over",
        "than",
        "then",
        "also",
    ]
)

# Hand-written concept groups. Words are stemmed with the same stemmer before lookup.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "concept_code": [
        "coding",
        "code",
        "programming",
        "program",
        "developer",
        "development",
        "dev",
        "software",
        "python",
        "java",
        "javascript",
        "engineer",
        "compile",
        "coder",
    ],
    "concept_travel": [
        "travel",
        "trip",
        "portable",
        "light",
        "lightweight",
        "ultralight",
        "ultraportable",
        "compact",
        "commute",
        "commuting",
        "commuter",
        "carry",
        "backpack",
        "mobile",
        "flight",
    ],
    "concept_study": [
        "study",
        "student",
        "school",
        "university",
        "homework",
        "class",
        "college",
        "learning",
    ],
    "concept_gaming": ["gaming", "game", "gamer", "esports", "fps", "play"],
    "concept_creative": [
        "video",
        "editing",
        "render",
        "rendering",
        "3d",
        "creative",
        "design",
        "designer",
        "photo",
        "animation",
        "content",
        "creator",
        "graphic",
    ],
    "concept_office": [
        "office",
        "business",
        "spreadsheet",
        "email",
        "document",
        "productivity",
        "desk",
        "home",
        "remote",
    ],
    "concept_browse": ["browsing", "web", "internet", "streaming", "casual"],
    "concept_heavy": ["workstation", "powerful", "performance", "cad"],
}


def stem(word: str) -> str:
    """Tiny deterministic suffix stripper (not linguistically complete)."""
    if len(word) <= 3:
        return word
    for suffix in ("ment", "ing", "ers", "er", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


_CONCEPT_OF: dict[str, list[str]] = {}
for _concept, _words in CONCEPTS.items():
    for _word in _words:
        _CONCEPT_OF.setdefault(stem(_word), []).append(_concept)


def concepts_of(text: str) -> frozenset[str]:
    """Concept tokens triggered by the words of a text (curated lexicon only)."""
    english = (concept for word in tokenize(text) for concept in _CONCEPT_OF.get(stem(word), ()))
    return frozenset(english) | persian.concepts_of(text)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@lru_cache(maxsize=65536)
def _bin_and_sign(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 40) & 1 else -1.0


def features(text: str) -> dict[str, float]:
    """Weighted sparse features: stems, concept tokens and character trigrams."""
    counts: dict[str, float] = {}
    # Persian phrases are translated to the catalog's own vocabulary first, so an exact word
    # match is possible; unknown Persian words contribute nothing.
    for word in [*tokenize(text), *tokenize(" ".join(persian.translate(text)))]:
        root = stem(word)
        counts[f"w:{root}"] = counts.get(f"w:{root}", 0.0) + WORD_WEIGHT
        for concept in _CONCEPT_OF.get(root, ()):
            counts[concept] = counts.get(concept, 0.0) + CONCEPT_WEIGHT
        if len(root) >= 4:
            padded = f"^{root}$"
            for i in range(len(padded) - 2):
                key = f"g:{padded[i : i + 3]}"
                counts[key] = counts.get(key, 0.0) + TRIGRAM_WEIGHT
    for concept in persian.concepts_of(text):  # Persian words map to the same concept ids
        counts[concept] = counts.get(concept, 0.0) + CONCEPT_WEIGHT
    return {k: 1.0 + math.log(v) if v >= 1 else v for k, v in counts.items()}


class LocalEmbedder:
    """Deterministic offline embedder. Fit once on the catalog, then encode products and queries."""

    model_id = MODEL_ID
    pinned = True  # the revision is a content hash of the fitted state, so it is immutable
    max_tokens = 512

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._idf: np.ndarray | None = None

    # --- state -----------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._idf is not None

    @property
    def revision(self) -> str:
        if self._idf is None:
            return "unfitted"
        digest = hashlib.sha256(self._idf.tobytes() + str(self.dim).encode()).hexdigest()
        return f"idf-{digest[:16]}"

    def fit(self, texts: list[str]) -> LocalEmbedder:
        """Learn per-bin inverse document frequency from the catalog documents."""
        if not texts:
            raise EmbeddingError("cannot fit on an empty corpus")
        doc_freq = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            doc_freq[list({_bin_and_sign(f, self.dim)[0] for f in features(text)})] += 1.0
        n = len(texts)
        self._idf = (np.log((1.0 + n) / (1.0 + doc_freq)) + 1.0).astype(np.float32)
        return self

    def state_files(self) -> dict[str, bytes]:
        if self._idf is None:
            raise EmbeddingError("embedder is not fitted")
        buffer = io.BytesIO()
        np.save(buffer, self._idf, allow_pickle=False)
        meta = {"model_id": MODEL_ID, "dim": self.dim, "revision": self.revision}
        return {
            STATE_JSON: json.dumps(meta, sort_keys=True).encode("utf-8"),
            STATE_NPY: buffer.getvalue(),
        }

    def load_state(self, files: dict[str, bytes]) -> None:
        """Restore fitted state saved with an index. Rejects anything that does not fit."""
        try:
            meta = json.loads(files[STATE_JSON])
            idf = np.load(io.BytesIO(files[STATE_NPY]), allow_pickle=False)
        except (KeyError, ValueError, OSError) as exc:
            raise EmbeddingError(f"embedder state is missing or unreadable: {exc}") from exc
        if meta.get("model_id") != MODEL_ID or meta.get("dim") != self.dim:
            raise EmbeddingError("embedder state does not match this embedder")
        if idf.shape != (self.dim,) or idf.dtype != np.float32 or not np.isfinite(idf).all():
            raise EmbeddingError("embedder state has an invalid IDF vector")
        self._idf = idf

    # --- encoding --------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(tokenize(text))

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for feature, weight in features(text).items():
            idx, sign = _bin_and_sign(feature, self.dim)
            idf = 1.0 if self._idf is None else float(self._idf[idx])
            vec[idx] += sign * weight * idf
        norm = np.linalg.norm(vec)
        if norm == 0:
            # No usable features (only stopwords or symbols): a fixed unit vector keeps the
            # contract and scores near zero against real documents.
            vec[0] = 1.0
            norm = 1.0
        return (vec / norm).astype(np.float32)

    def encode_products(self, texts: list[str]) -> np.ndarray:
        matrix = np.ascontiguousarray([self._encode(t) for t in texts], dtype=np.float32)
        return validate_vectors(matrix)

    def encode_query(self, text: str) -> np.ndarray:
        return validate_vectors(np.ascontiguousarray([self._encode(text)], dtype=np.float32))
