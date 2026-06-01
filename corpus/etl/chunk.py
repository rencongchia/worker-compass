"""
Chunk raw scraped documents into overlapping word-bounded segments.

Flow: corpus/raw/{source}.jsonl → corpus/chunks/{source}.jsonl

Token approximation: 1 word ≈ 1.33 tokens for English legal text.
  500 tokens ≈ 375 words (WORDS_PER_CHUNK)
   50 tokens ≈  37 words (OVERLAP_WORDS)

Each chunk inherits all metadata from its parent document and prepends
the section heading so that every chunk is self-contained for retrieval.
"""

import json
import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

WORDS_PER_CHUNK = 375
OVERLAP_WORDS = 37

RAW_DIR = Path("corpus/raw")
CHUNKS_DIR = Path("corpus/chunks")


def chunk_document(doc: dict) -> list[dict]:
    text = doc.get("text", "").strip()
    if not text:
        return []

    heading = _build_heading(doc)
    words = text.split()
    today = date.today().isoformat()

    if len(words) <= WORDS_PER_CHUNK:
        return [_make_chunk(doc, heading + text, 0, today)]

    chunks = []
    idx = 0
    start = 0
    while start < len(words):
        end = min(start + WORDS_PER_CHUNK, len(words))
        chunk_text = heading + " ".join(words[start:end])
        chunks.append(_make_chunk(doc, chunk_text, idx, today))
        idx += 1
        if end >= len(words):
            break
        start = end - OVERLAP_WORDS

    return chunks


def _build_heading(doc: dict) -> str:
    parts = [p for p in [doc.get("act_name"), doc.get("section_title")] if p]
    return " — ".join(parts) + "\n\n" if parts else ""


def _make_chunk(doc: dict, text: str, idx: int, today: str) -> dict:
    return {
        "chunk_id": f"{doc['doc_id']}_c{idx}",
        "doc_id": doc["doc_id"],
        "source": doc.get("source", ""),
        "act_name": doc.get("act_name", ""),
        "section_id": doc.get("section_id", ""),
        "section_title": doc.get("section_title", ""),
        "url": doc.get("url", ""),
        "date_retrieved": doc.get("date_retrieved", today),
        "corpus_snapshot_date": today,
        "text": text,
        "word_count": len(text.split()),
    }


def chunk_source(source_name: str) -> int:
    raw_path = RAW_DIR / f"{source_name}.jsonl"
    if not raw_path.exists():
        log.warning("  %s not found — skipping", raw_path)
        return 0

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / f"{source_name}.jsonl"

    total = 0
    with raw_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for chunk in chunk_document(doc):
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total += 1

    log.info("  [chunk] %s → %d chunks", source_name, total)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for src in ["mom", "tadm", "tafep"]:
        chunk_source(src)
