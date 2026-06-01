"""
Shared utilities for finding, downloading, and parsing PDFs linked from scraped pages.

Used by all four scrapers to supplement HTML content with PDF attachments
(fee cap tables, guidance booklets, pass conditions, etc.) that government sites
frequently publish as PDF-only documents.
"""

import logging
import re
from io import BytesIO
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def find_pdf_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Return deduplicated absolute URLs for all PDF links on the page."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf") or "viewtype=pdf" in href.lower():
            links.append(urljoin(base_url, href))
    return list(dict.fromkeys(links))


def download_pdf(session: requests.Session, url: str) -> bytes | None:
    """Download a URL and return bytes if it resolves to a PDF, else None."""
    try:
        r = session.get(url, timeout=60)
        content_type = r.headers.get("Content-Type", "")
        if r.status_code == 200 and (
            "pdf" in content_type.lower() or url.lower().endswith(".pdf")
        ):
            return r.content
        log.debug("  %s → %s %s (not a PDF — skipping)", url, r.status_code, content_type[:40])
        return None
    except Exception as exc:
        log.warning("  PDF download failed %s: %s", url, exc)
        return None


def parse_pdf_to_docs(
    pdf_bytes: bytes,
    pdf_url: str,
    source: str,
    act_name: str,
    today: str,
    id_prefix: str,
) -> list[dict]:
    """
    Parse PDF bytes into section-level documents.

    Uses pymupdf4llm to convert PDF pages to Markdown, then splits on
    Markdown headings (## / ###) to produce one doc per logical section.
    Falls back to a single full-text document if no headings are found.
    """
    try:
        import fitz
        import pymupdf4llm
    except ImportError:
        log.warning("pymupdf4llm / fitz not installed — skipping PDF %s", pdf_url)
        return []

    try:
        with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc_pdf:
            # page_chunks=True gives one dict per page with metadata.page (0-indexed)
            page_chunks = pymupdf4llm.to_markdown(doc_pdf, page_chunks=True)
    except Exception as exc:
        log.warning("  PDF parse error %s: %s", pdf_url, exc)
        return []

    safe_prefix = re.sub(r"[^a-z0-9]+", "_", id_prefix.lower())[:30]
    docs = []

    for page_info in page_chunks:
        page_text = page_info.get("text", "").strip()
        if len(page_text) < 80:
            continue
        # pymupdf4llm returns 0-indexed page numbers in metadata
        page_num = page_info.get("metadata", {}).get("page", 0) + 1
        first_line = page_text.splitlines()[0]
        heading = re.sub(r"^#+\s*", "", first_line).strip()

        docs.append({
            "doc_id": f"{safe_prefix}_p{page_num}",
            "source": source,
            "act_name": act_name,
            "section_id": str(page_num),
            "section_title": heading or f"Page {page_num}",
            "url": f"{pdf_url}#page={page_num}",
            "page_number": page_num,
            "date_retrieved": today,
            "text": page_text,
        })

    # Fallback: if page_chunks gave no usable pages, use full-doc markdown
    if not docs:
        md_text = "\n\n".join(p.get("text", "") for p in page_chunks).strip()
        if md_text:
            docs.append({
                "doc_id": f"{safe_prefix}_pdf_full",
                "source": source,
                "act_name": act_name,
                "section_id": "full",
                "section_title": "Full document",
                "url": pdf_url,
                "date_retrieved": today,
                "text": md_text,
            })

    return docs


def scrape_linked_pdfs(
    soup: BeautifulSoup,
    page_url: str,
    session: requests.Session,
    source: str,
    act_name: str,
    today: str,
    id_prefix: str,
) -> list[dict]:
    """
    Find all PDF links on a page, download each, and return parsed docs.

    Call this after successfully scraping an HTML page to pick up any
    PDF attachments (fee tables, guidance booklets, pass conditions).
    """
    pdf_links = find_pdf_links(soup, page_url)
    if not pdf_links:
        return []

    all_docs = []
    for pdf_url in pdf_links:
        log.info("    PDF linked: %s", pdf_url)
        pdf_bytes = download_pdf(session, pdf_url)
        if pdf_bytes is None:
            continue
        # Use last path component as part of the id prefix
        slug = re.sub(r"[^a-z0-9]+", "_", pdf_url.split("/")[-1].lower().replace(".pdf", ""))[:20]
        docs = parse_pdf_to_docs(pdf_bytes, pdf_url, source, act_name, today, f"{id_prefix}_{slug}")
        log.info("      → %d sections from PDF", len(docs))
        all_docs.extend(docs)

    return all_docs
