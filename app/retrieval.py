"""
Hybrid BM25 + BGE-M3 retrieval with Reciprocal Rank Fusion (RRF).

Pipeline:
  query (English) → BM25 top-N + BGE-M3 dense top-N → RRF → top_k chunks
  top-1 dense cosine similarity → refusal gate threshold check in agent.py

RRF formula: score(d) = Σ 1 / (k + rank_i(d))  where k=60 (standard)
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RRF_K = 60


@dataclass
class RetrievalResult:
    chunks: list[dict]
    top_similarity: float # cosine similarity of best dense match (0–1)
    bm25_hits: int
    dense_hits: int
    chunk_similarities: dict = field(default_factory=dict) # chunk_id → cosine_sim


class HybridRetriever:
    def __init__(self, bm25_path: str, lancedb_path: str, top_k: int = 8,
                 bm25_candidates: int = 20, dense_candidates: int = 20):
        self.top_k = top_k
        self.bm25_candidates = bm25_candidates
        self.dense_candidates = dense_candidates

        self._bm25 = None
        self._bm25_chunks: list[dict] = []
        self._table = None
        self._embed_model = None

        self._bm25_path = Path(bm25_path)
        self._lancedb_path = Path(lancedb_path)

    # ------------------------------------------------------------------
    # Lazy loading — deferred so Streamlit doesn't block on cold import
    # ------------------------------------------------------------------

    def _load_bm25(self) -> None:
        if self._bm25 is not None:
            return
        if not self._bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self._bm25_path}. "
                "Run: python corpus/etl/run_all.py && python corpus/build_index.py"
            )
        with self._bm25_path.open("rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._bm25_chunks = data["chunks"]
        log.info("BM25 index loaded: %d docs", len(self._bm25_chunks))

    def _load_lancedb(self) -> None:
        if self._table is not None:
            return
        import lancedb
        db = lancedb.connect(str(self._lancedb_path))
        self._table = db.open_table("chunks")
        log.info("LanceDB table opened")

    def _load_embedder(self) -> None:
        if self._embed_model is not None:
            return
        import warnings
        from sentence_transformers import SentenceTransformer
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torchvision.*")
            self._embed_model = SentenceTransformer("BAAI/bge-m3")
        log.info("BGE-M3 embedder loaded")

    def _embed(self, text: str) -> list[float]:
        self._load_embedder()
        vec = self._embed_model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str) -> list[tuple[dict, float]]:
        """Return (chunk, bm25_score) sorted descending."""
        self._load_bm25()
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][: self.bm25_candidates]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._bm25_chunks[idx], float(scores[idx])))
        return results

    def _dense_search(self, query: str) -> list[tuple[dict, float]]:
        """Return (chunk, cosine_similarity) sorted descending."""
        self._load_lancedb()
        vec = self._embed(query)
        rows = (
            self._table.search(vec)
            .metric("cosine")
            .limit(self.dense_candidates)
            .to_list()
        )
        # LanceDB cosine metric returns distance (0 = identical, 2 = opposite)
        # cosine_similarity = 1 - distance  (for unit-normalised vectors, dist ∈ [0,2])
        results = []
        for row in rows:
            sim = 1.0 - float(row.get("_distance", 1.0))
            chunk = {k: v for k, v in row.items() if k != "_distance"}
            results.append((chunk, sim))
        return results

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        bm25_results = self._bm25_search(query)
        dense_results = self._dense_search(query)

        top_similarity = dense_results[0][1] if dense_results else 0.0

        # Build per-chunk similarity map from dense results
        dense_sim_map: dict[str, float] = {
            chunk["chunk_id"]: sim for chunk, sim in dense_results
        }

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for rank, (chunk, _) in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunk_map[cid] = chunk

        for rank, (chunk, _) in enumerate(dense_results):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunk_map[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_chunks = [chunk_map[cid] for cid, _ in ranked[: self.top_k]]

        return RetrievalResult(
            chunks=top_chunks,
            top_similarity=top_similarity,
            bm25_hits=len(bm25_results),
            dense_hits=len(dense_results),
            chunk_similarities=dense_sim_map,
        )
