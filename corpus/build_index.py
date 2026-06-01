"""
Build BM25 and LanceDB indexes from chunked corpus.

Usage (from project root):
    python corpus/build_index.py

Inputs:  corpus/chunks/*.jsonl
Outputs:
  corpus/bm25_index.pkl  — BM25Okapi index + chunk list (for keyword retrieval)
  corpus/lancedb/        — LanceDB table with BGE-M3 embeddings (for dense retrieval)

Embedding model: BAAI/bge-m3 via sentence-transformers (1024d, 119 languages)
LanceDB table name: "chunks"

This script is re-entrant — running it again overwrites the existing indexes.
On CPU, expect ~2-3 minutes per 1000 chunks. BGE-M3 model weights are volume-mounted into the container at runtime
(see docker-compose.yml). Download once, reuse across rebuilds.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

log = logging.getLogger(__name__)

CHUNKS_DIR = Path("corpus/chunks")
LANCEDB_DIR = Path("corpus/lancedb")
BM25_PATH = Path("corpus/bm25_index.pkl")
EMBED_BATCH = 32


def load_chunks() -> list[dict]:
    chunks = []
    for jsonl_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    log.info("Loaded %d chunks from %s", len(chunks), CHUNKS_DIR)
    return chunks


def build_bm25(chunks: list[dict]) -> None:
    from rank_bm25 import BM25Okapi

    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    log.info("BM25 index saved → %s", BM25_PATH)


def build_lancedb(chunks: list[dict]) -> None:
    import lancedb
    import numpy as np
    from sentence_transformers import SentenceTransformer

    log.info("Loading BAAI/bge-m3 model …")
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")

    LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(LANCEDB_DIR))

    texts = [c["text"] for c in chunks]
    log.info("Embedding %d chunks in batches of %d …", len(texts), EMBED_BATCH)

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(vecs.tolist())
        if (i // EMBED_BATCH) % 10 == 0:
            log.info("  %d / %d chunks embedded", min(i + EMBED_BATCH, len(texts)), len(texts))

    records = []
    for chunk, vec in zip(chunks, all_embeddings):
        records.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source": chunk["source"],
            "act_name": chunk["act_name"],
            "section_id": chunk["section_id"],
            "section_title": chunk["section_title"],
            "url": chunk["url"],
            "date_retrieved": chunk["date_retrieved"],
            "corpus_snapshot_date": chunk["corpus_snapshot_date"],
            "text": chunk["text"],
            "vector": vec,
        })

    # Overwrite existing table so this script is idempotent
    table = db.create_table("chunks", data=records, mode="overwrite")
    log.info("LanceDB table 'chunks' created: %d rows → %s", len(records), LANCEDB_DIR)

    # Create an ANN index for fast approximate search (IVF_PQ)
    table.create_index(
        metric="cosine",
        num_partitions=min(256, max(1, len(records) // 40)),
        num_sub_vectors=32,
    )
    log.info("ANN index created")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not CHUNKS_DIR.exists() or not any(CHUNKS_DIR.glob("*.jsonl")):
        log.error("No chunks found in %s — run corpus/etl/run_all.py first", CHUNKS_DIR)
        sys.exit(1)

    chunks = load_chunks()
    if not chunks:
        log.error("All chunk files are empty")
        sys.exit(1)

    log.info("--- Building BM25 index ---")
    build_bm25(chunks)

    log.info("--- Building LanceDB index ---")
    build_lancedb(chunks)

    log.info("=== Indexing complete ===")
    log.info("  BM25:    %s", BM25_PATH)
    log.info("  LanceDB: %s", LANCEDB_DIR)


if __name__ == "__main__":
    main()
