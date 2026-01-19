import os
import json
from typing import List, Tuple

import numpy as np

FAISS_DIR = os.getenv("FAISS_DIR", "/data/faiss").strip()
INDEX_PATH = os.path.join(FAISS_DIR, "docs.index")
IDS_PATH = os.path.join(FAISS_DIR, "docs_ids.jsonl")

# Lazy-loaded globals
_g_index = None
_g_ids: List[str] = []
_g_dim: int | None = None


def _require_files():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"FAISS index missing: {INDEX_PATH}")
    if not os.path.exists(IDS_PATH):
        raise RuntimeError(f"FAISS id map missing: {IDS_PATH}")


def load_faiss():
    global _g_index, _g_ids, _g_dim
    if _g_index is not None:
        return _g_index, _g_ids, _g_dim

    _require_files()

    import faiss  # faiss-cpu

    _g_index = faiss.read_index(INDEX_PATH)

    ids: List[str] = []
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(obj["document_id"])
    _g_ids = ids
    # sanity check: id-map should cover index rows
    if _g_index.ntotal > len(_g_ids):
        raise RuntimeError(f"FAISS id map smaller than index: ntotal={_g_index.ntotal} ids={len(_g_ids)}")

    _g_dim = _g_index.d
    return _g_index, _g_ids, _g_dim


def search_centroid(vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
    """
    vec: shape (dim,), MUST be float32 and L2-normalized.
    returns list of (document_id, score) sorted by score desc
    """
    index, ids, dim = load_faiss()
    if vec.shape != (dim,):
        raise ValueError(f"centroid dim mismatch: got {vec.shape}, expected {(dim,)}")

    # ✅ enforce cosine expectation
    n = float(np.linalg.norm(vec))
    if not (0.99 <= n <= 1.01):
        raise ValueError(f"centroid must be L2-normalized (norm={n:.4f})")

    q = vec.astype("float32")[None, :]
    D, I = index.search(q, k)  # D: scores, I: indices
    out: List[Tuple[str, float]] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        if idx >= len(ids):
            continue
        out.append((ids[idx], float(score)))
    return out
