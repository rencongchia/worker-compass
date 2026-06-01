"""
Scrape TADM guidance pages (tal.sg/tadm) and the ECT Claimant Guide PDF
from the Singapore Judiciary website.

Licensing: TAL/TADM content is not under an open licence. This prototype
is a non-commercial research submission under fair dealing provisions.

Sources:
  tal.sg/tadm — mediation guide, fees, deadlines, process overview
  judiciary.gov.sg — Employment Claims Tribunal Claimant Guide (PDF)

Output: corpus/raw/tadm.jsonl  (one JSON object per page or PDF section)
"""

import json
import logging
import re
import time
from datetime import date
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from corpus.etl.pdf_utils import scrape_linked_pdfs

log = logging.getLogger(__name__)

RAW_DIR = Path("corpus/raw")
TODAY = date.today().isoformat()
DELAY = 1.5

TADM_PAGES = [
    {
        "url": "https://www.tal.sg/tadm",
        "title": "TADM Overview",
        "topic": "TADM",
    },
    {
        "url": "https://www.tal.sg/tadm/know-your-options",
        "title": "TADM — Know Your Options in an Employment Dispute",
        "topic": "TADM",
    },
    {
        "url": "https://www.tal.sg/tadm/mediation-guide-3",
        "title": "TADM — Mediation Guide for Salary-Related Claims",
        "topic": "TADM Mediation",
    },
    {
        "url": "https://www.tal.sg/tadm/faqs/advisory/employment-dispute",
        "title": "TADM — Employment Dispute FAQ",
        "topic": "TADM",
    },
    {
        "url": "https://www.tal.sg/tadm/eservices/employees-file-employment-claim",
        "title": "TADM — How to File an Employment Claim",
        "topic": "TADM Mediation",
    },
    {
        "url": "https://www.tal.sg/tadm/about",
        "title": "TADM — About (including ECT)",
        "topic": "ECT",
    },
]

ECT_PDF_URLS = [
    "https://www.judiciary.gov.sg/docs/default-source/civil-docs/ect_guide.pdf",
    "https://www.judiciary.gov.sg/docs/default-source/civil-docs/cjts_guide_to_filing_ect.pdf",
    "https://www.judiciary.gov.sg/docs/default-source/default-document-library/ect-guide-for-claimants.pdf",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "WorkerCompass/0.1 research prototype (non-commercial)",
    "Accept": "text/html,application/xhtml+xml,application/pdf",
    "Accept-Language": "en-SG,en;q=0.9",
})


def _get_html(url: str) -> BeautifulSoup | None:
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
    for tag in soup.find_all(class_=["breadcrumb", "sidebar", "share", "feedback"]):
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


def _scrape_tadm_pages(fout) -> int:
    total = 0
    for page in TADM_PAGES:
        log.info("  TADM: %s", page["title"])
        soup = _get_html(page["url"])
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
            "doc_id": f"tadm_{safe_id}",
            "source": "tadm",
            "act_name": f"TADM Guidance — {page['topic']}",
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
            source="tadm",
            act_name=f"TADM Guidance — {page['topic']}",
            today=TODAY,
            id_prefix=f"tadm_{safe_id}",
        )
        for pdoc in pdf_docs:
            fout.write(json.dumps(pdoc, ensure_ascii=False) + "\n")
        total += len(pdf_docs)

        time.sleep(DELAY)
    return total


def _scrape_ect_pdf(fout) -> int:
    """
    Download the ECT Claimant Guide PDF and extract text with pymupdf4llm.
    pymupdf4llm converts PDF pages to Markdown, preserving headings and tables.
    Each logical section of the guide becomes a separate document.
    """
    try:
        import pymupdf4llm
        import fitz  # PyMuPDF
    except ImportError:
        log.warning("pymupdf4llm not installed — skipping ECT PDF")
        return 0

    pdf_bytes: bytes | None = None
    pdf_url: str = ""

    for url in ECT_PDF_URLS:
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                pdf_bytes = r.content
                pdf_url = url
                log.info("  ECT PDF: downloaded %d bytes from %s", len(pdf_bytes), url)
                break
            else:
                log.debug("  %s → %s (not a PDF)", url, r.status_code)
        except Exception as exc:
            log.warning("  %s failed: %s", url, exc)
        time.sleep(DELAY)

    if pdf_bytes is None:
        log.warning("  ECT PDF not found at any known URL — skipping")
        log.warning("  Download manually from judiciary.gov.sg and place at corpus/raw/ect_guide.pdf")
        # Try local fallback
        local = RAW_DIR / "ect_guide.pdf"
        if local.exists():
            log.info("  Found local %s", local)
            pdf_bytes = local.read_bytes()
            pdf_url = str(local)
        else:
            return 0

    # Extract markdown from PDF
    with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc_pdf:
        md_text = pymupdf4llm.to_markdown(doc_pdf)

    # Split on markdown headings (## or ###) to create section-level documents
    sections = re.split(r"\n(?=#{1,3} )", md_text)
    total = 0
    for idx, section in enumerate(sections):
        section = section.strip()
        if len(section) < 80:
            continue
        # Extract heading from first line
        first_line = section.splitlines()[0]
        heading = re.sub(r"^#+\s*", "", first_line).strip()
        body = "\n".join(section.splitlines()[1:]).strip()
        text = f"{heading}\n\n{body}" if body else section

        doc = {
            "doc_id": f"ect_guide_s{idx}",
            "source": "tadm",
            "act_name": "ECT Claimant Guide",
            "section_id": str(idx),
            "section_title": heading or f"Section {idx}",
            "url": pdf_url,
            "date_retrieved": TODAY,
            "text": text,
        }
        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
        total += 1

    log.info("  ECT PDF: %d sections extracted", total)
    return total


def scrape() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "tadm.jsonl"
    total = 0

    with out_path.open("w", encoding="utf-8") as fout:
        total += _scrape_tadm_pages(fout)
        total += _scrape_ect_pdf(fout)

    log.info("TADM done: %d documents → %s", total, out_path)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scrape()
