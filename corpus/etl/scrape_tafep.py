"""
Scrape TAFEP (Tripartite Alliance for Fair and Progressive Employment Practices)
guidance pages on workplace fairness, discrimination, and the Workplace Fairness Act.

Licensing: TAFEP/TAL content is not under an open licence. This prototype
is a non-commercial research submission under fair dealing provisions.

Source: tal.sg/tafep

Output: corpus/raw/tafep.jsonl  (one JSON object per page)
"""

import json
import logging
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from corpus.etl.pdf_utils import scrape_linked_pdfs

log = logging.getLogger(__name__)

RAW_DIR = Path("corpus/raw")
TODAY = date.today().isoformat()
DELAY = 1.5

TAFEP_PAGES = [
    {
        "url": "https://www.tal.sg/tafep",
        "title": "TAFEP Overview",
        "topic": "Workplace Fairness",
    },
    {
        "url": "https://www.tal.sg/tafep/workplace-fairness",
        "title": "Workplace Fairness Act",
        "topic": "Workplace Fairness Act",
    },
    {
        "url": "https://www.tal.sg/tafep/workplace-fairness/individuals",
        "title": "Workplace Fairness Act — Guide for Individuals",
        "topic": "Workplace Fairness Act",
    },
    {
        "url": "https://www.tal.sg/tafep/employees",
        "title": "TAFEP — Employee Guidance",
        "topic": "Workplace Fairness",
    },
    {
        "url": "https://www.tal.sg/tafep/employees/discrimination",
        "title": "TAFEP — Workplace Discrimination",
        "topic": "Discrimination",
    },
    {
        "url": "https://www.tal.sg/tafep/employees/fair-recruitment",
        "title": "TAFEP — Fair Recruitment Practices",
        "topic": "Recruitment",
    },
    {
        "url": "https://www.tal.sg/tafep/getting-started/fair/tripartite-guidelines",
        "title": "Tripartite Guidelines on Fair Employment Practices (TGFEP)",
        "topic": "TGFEP",
    },
    {
        "url": "https://www.tal.sg/tafep/contact-us",
        "title": "TAFEP — Report Discrimination / Getting Help",
        "topic": "Help Resources",
    },
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "WorkerCompass/0.1 research prototype (non-commercial)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-SG,en;q=0.9",
})


def _get(url: str) -> BeautifulSoup | None:
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception as exc:
        log.warning("GET %s failed: %s", url, exc)
        return None


def _extract_content(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    for tag in soup.find_all(class_=["breadcrumb", "sidebar", "share",
                                      "feedback", "related-content", "pagination"]):
        tag.decompose()

    content = (
        soup.find("div", class_=lambda c: c and "content" in c.lower())
        or soup.find("article")
        or soup.find("main")
        or soup.find("body")
    )
    if not content:
        return ""

    lines = [ln.strip() for ln in content.get_text(separator="\n").splitlines() if ln.strip()]
    return "\n".join(lines)


def scrape() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "tafep.jsonl"
    total = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for page in TAFEP_PAGES:
            log.info("  TAFEP: %s", page["title"])
            soup = _get(page["url"])
            if soup is None:
                time.sleep(DELAY)
                continue

            text = _extract_content(soup)
            if len(text) < 100:
                log.warning("    Too short — skipping")
                time.sleep(DELAY)
                continue

            safe_id = re.sub(r"[^a-z0-9]+", "_", page["title"].lower())[:40]
            doc = {
                "doc_id": f"tafep_{safe_id}",
                "source": "tafep",
                "act_name": f"TAFEP Guidance — {page['topic']}",
                "section_id": "",
                "section_title": page["title"],
                "url": page["url"],
                "date_retrieved": TODAY,
                "text": text,
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total += 1

            # Also download any PDFs linked from the page
            pdf_docs = scrape_linked_pdfs(
                soup, page["url"], SESSION,
                source="tafep",
                act_name=f"TAFEP Guidance — {page['topic']}",
                today=TODAY,
                id_prefix=f"tafep_{safe_id}",
            )
            for pdoc in pdf_docs:
                fout.write(json.dumps(pdoc, ensure_ascii=False) + "\n")
            total += len(pdf_docs)

            time.sleep(DELAY)

    log.info("TAFEP done: %d documents → %s", total, out_path)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scrape()
