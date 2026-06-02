from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

EMBED_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(MODEL_NAME)


def _try_embed(texts: Sequence[str]) -> list[np.ndarray] | None:
    try:
        return [np.asarray(v, dtype=np.float32) for v in _embedder().embed(list(texts))]
    except Exception as exc:
        LOGGER.warning("fastembed unavailable, vectors will be skipped: %s", exc)
        return None


def pack(arr: np.ndarray) -> bytes:
    a = np.asarray(arr, dtype=np.float32)
    if a.shape != (EMBED_DIM,):
        raise ValueError(f"vector must be shape ({EMBED_DIM},), got {a.shape}")
    norm = float(np.linalg.norm(a))
    if norm > 0.0:
        a = a / norm
    return a.tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def embed(text: str) -> bytes | None:
    vecs = _try_embed([text])
    if not vecs:
        return None
    return pack(vecs[0])


def embed_batch(texts: list[str]) -> list[bytes | None]:
    if not texts: return []
    vecs = _try_embed(texts)
    if vecs is None: return [None] * len(texts)
    return [pack(v) for v in vecs]


# Neutral unit centroid — used as fallback momentum anchor when no query vec available.
def generic_centroid() -> np.ndarray | None:
    try:
        v = np.ones(EMBED_DIM, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else None
    except Exception: return None
