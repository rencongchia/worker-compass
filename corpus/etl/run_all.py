"""
Run all corpus ETL scrapers and then chunk the output.

Usage (from project root):
    python corpus/etl/run_all.py

After this completes, run:
    python corpus/build_index.py

The full pipeline:
  run_all.py  → corpus/raw/*.jsonl  → corpus/chunks/*.jsonl
  build_index.py → corpus/lancedb/ + corpus/bm25_index.pkl
"""

import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from corpus.etl import chunk, scrape_mom, scrape_tadm, scrape_tafep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SOURCES = [
    ("mom", scrape_mom),
    ("tadm", scrape_tadm),
    ("tafep", scrape_tafep),
]


def main() -> None:
    log.info("=== WorkerCompass Corpus ETL ===")

    # Step 1: Scrape
    log.info("--- Step 1: Scraping ---")
    total_docs = 0
    for name, module in SOURCES:
        log.info("Scraping %s …", name)
        try:
            n = module.scrape()
            total_docs += n
            log.info("  %s: %d documents", name, n)
        except Exception as exc:
            log.error("  %s scrape failed: %s — continuing", name, exc)

    log.info("Total raw documents: %d", total_docs)

    # Step 2: Chunk
    log.info("--- Step 2: Chunking ---")
    total_chunks = 0
    for name, _ in SOURCES:
        n = chunk.chunk_source(name)
        total_chunks += n

    log.info("Total chunks: %d", total_chunks)
    log.info("=== ETL complete. Run: python corpus/build_index.py ===")


if __name__ == "__main__":
    main()
