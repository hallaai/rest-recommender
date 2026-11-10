"""Item text embeddings with three interchangeable backends, so cost and quality can be compared:

  gemini  Gemini API (paid per token)         768-d
  bge     BAAI/bge-base-en-v1.5 on local CPU  768-d, free, needs the model in the HF cache
  tfidf   TF-IDF + truncated SVD               128-d, free, seconds

    python -m recsys.embed --backend bge

Writes data/embeddings/<backend>.npy (float32, L2-normalised, row = item_id) and
<backend>.json (dimension, seconds, USD). Rows follow item_id order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, load_settings, quiet_http_logs
from .synth.vocab import CUISINES

log = logging.getLogger(__name__)
EMB_DIR = DATA_DIR / "embeddings"
BACKENDS = ("gemini", "bge", "tfidf")
_LABEL = {c.key: c.label for c in CUISINES}


def item_texts(items: pd.DataFrame, restaurants: pd.DataFrame) -> list[str]:
    """One descriptive line per item: dish, description, cuisine, category, diet, restaurant."""
    r = restaurants.set_index("restaurant_id")
    out = []
    for it in items.itertuples():
        rr = r.loc[it.restaurant_id]
        diet = ", ".join(x for x, f in (("vegetarian", it.vegetarian), ("vegan", it.vegan),
                                         ("spicy", it.spicy)) if f)
        tags = ", ".join(it.tags) if len(it.tags) else ""
        out.append(f"{it.name}. {it.description} {_LABEL[rr.cuisine]} {it.category}"
                   f"{' (' + diet + ')' if diet else ''}. {tags}. From {rr['name']}.".strip())
    return out


def file_hash(path: Path) -> str:
    """Short content hash: embeddings are stale when the dishes they were built from change."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


GEMINI_BATCH = 50   # the embed API caps input tokens per request; long dish texts need small batches


def _embed_batch(g, texts: list[str]) -> list[list[float]]:
    """Embed one batch; on failure, split in half and retry, down to single texts."""
    from .llm import LLMError
    try:
        return g.embed(texts, purpose="item_embeddings")
    except LLMError:
        if len(texts) == 1:
            raise
        mid = len(texts) // 2
        log.warning("  gemini batch of %d failed; splitting", len(texts))
        return _embed_batch(g, texts[:mid]) + _embed_batch(g, texts[mid:])


def embed_gemini(texts: list[str]) -> tuple[np.ndarray, float]:
    from .llm import Gemini
    g = Gemini(load_settings())
    before = sum(r.usd for r in g.ledger.records())
    vecs: list[list[float]] = []
    for i in range(0, len(texts), GEMINI_BATCH):
        vecs += _embed_batch(g, texts[i:i + GEMINI_BATCH])
        if (i // GEMINI_BATCH) % 10 == 0:
            log.info("  gemini %d/%d", min(i + GEMINI_BATCH, len(texts)), len(texts))
    return normalise(np.array(vecs)), sum(r.usd for r in g.ledger.records()) - before


def embed_bge(texts: list[str]) -> tuple[np.ndarray, float]:
    import os
    from huggingface_hub import try_to_load_from_cache
    if try_to_load_from_cache("BAAI/bge-base-en-v1.5", "config.json"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")      # cached: skip slow online checks
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    return normalise(model.encode(texts, batch_size=64, show_progress_bar=False)), 0.0


def embed_tfidf(texts: list[str], dim: int = 128) -> tuple[np.ndarray, float]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    x = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    return normalise(TruncatedSVD(n_components=dim, random_state=0).fit_transform(x)), 0.0


def build(backend: str, data_dir: Path = DATA_DIR) -> dict:
    items = pd.read_parquet(data_dir / "items.parquet")
    rest = pd.read_parquet(data_dir / "restaurants.parquet")
    texts = item_texts(items, rest)
    t0 = time.perf_counter()
    fn = {"gemini": embed_gemini, "bge": embed_bge, "tfidf": embed_tfidf}[backend]
    vecs, usd = fn(texts)
    meta = {"backend": backend, "items": len(texts), "dim": int(vecs.shape[1]),
            "seconds": round(time.perf_counter() - t0, 2), "usd": round(usd, 4),
            "items_hash": file_hash(data_dir / "items.parquet"), "texts_example": texts[:3]}
    out = data_dir / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{backend}.npy", vecs)
    (out / f"{backend}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load(backend: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    path = data_dir / "embeddings" / f"{backend}.npy"
    if not path.exists():
        raise FileNotFoundError(f"No {backend} embeddings; run python -m recsys.embed --backend {backend}")
    return np.load(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, nargs="+", default=["tfidf", "bge"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_http_logs()
    for b in args.backend:
        log.info("Embedding with %s ...", b)
        log.info("  %s", build(b))


if __name__ == "__main__":
    main()
