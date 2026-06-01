"""
Scrape MOM topical guidance pages (mom.gov.sg) and ask.gov.sg Q&A pairs.

Licensing: MOM Terms of Use (clause 7) reserves all rights but permits fair
dealing for private study and research. This prototype is a non-commercial
research submission. Production deployment requires explicit MOM permission.

Sources:
  mom.gov.sg — topical guidance pages on WICA, salary, EA fees, repatriation,
                change of employer, work permit conditions
  ask.gov.sg  — structured Q→A pairs submitted to MOM

Output: corpus/raw/mom.jsonl  (one JSON object per page or Q&A pair)
"""

import json
import logging
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

# MOM topical guidance pages — scope is employment rights for work permit holders
MOM_PAGES = [
    # Salary & disputes
    {
        "url": "https://www.mom.gov.sg/employment-practices/salary",
        "topic": "Salary",
        "subtopic": "Overview",
    },
    {
        "url": "https://www.mom.gov.sg/employment-practices/managing-employment-disputes",
        "topic": "Salary",
        "subtopic": "Getting help for salary problems (TADM/ECT)",
    },
    {
        "url": "https://www.mom.gov.sg/employment-practices/salary/salary-deductions",
        "topic": "Salary",
        "subtopic": "Salary deductions",
    },
    # WICA
    {
        "url": "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation",
        "topic": "WICA",
        "subtopic": "Overview",
    },
    {
        "url": "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/what-is-wica",
        "topic": "WICA",
        "subtopic": "What WICA covers",
    },
    {
        "url": "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/types-of-compensation",
        "topic": "WICA",
        "subtopic": "Compensation and benefits",
    },
    {
        "url": "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/eligible-claims",
        "topic": "WICA",
        "subtopic": "Eligible claims and making a claim",
    },
    # Employment agencies
    {
        "url": "https://www.mom.gov.sg/employment-agencies/key-facts",
        "topic": "Employment Agencies",
        "subtopic": "Key facts and fee limits",
    },
    # Change of employer (hiring an existing CMP worker = worker-initiated transfer)
    {
        "url": "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/sector-specific-rules/hiring-existing-worker-in-construction-sector",
        "topic": "Change of Employer",
        "subtopic": "Change of employer — construction sector",
    },
    # Work permit conditions (covers repatriation obligations and worker duties)
    {
        "url": "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/sector-specific-rules/work-permit-conditions",
        "topic": "Work Permit",
        "subtopic": "Work permit conditions and repatriation obligations",
    },
    {
        "url": "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/sector-specific-rules/construction-sector-requirements",
        "topic": "Work Permit",
        "subtopic": "Construction sector work permit requirements",
    },
]

# ask.gov.sg topic pages for MOM Q&A
ASK_GOV_TOPICS = [
    "https://ask.gov.sg/mom",
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


def _extract_main_content(soup: BeautifulSoup) -> str:
    """
    Extract main page text from a MOM/gov.sg page.

    MOM pages (as of 2025) have their content in:
      - <div class="col-md-9 main-content"> or similar
      - Or in <article> / <main> tags
      - Or in <div class="field-items">

    Navigation, breadcrumbs, sidebars, and footers are excluded by
    removing them before extracting text.
    """
    # Remove noise elements first
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    for tag in soup.find_all(class_=["breadcrumb", "sidebar", "pagination",
                                      "related", "share", "feedback-form"]):
        tag.decompose()

    content = (
        soup.find("div", class_=lambda c: c and "main-content" in c)
        or soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=lambda c: c and "content" in c)
        or soup.find("div", id=lambda i: i and "content" in i)
    )
    if not content:
        content = soup.find("body") or soup

    text = content.get_text(separator="\n", strip=True)
    # Collapse excessive whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _scrape_mom_pages(fout) -> int:
    total = 0
    for page in MOM_PAGES:
        log.info("  MOM: %s — %s", page["topic"], page["subtopic"])
        soup = _get(page["url"])
        if soup is None:
            time.sleep(DELAY)
            continue

        # Try to get the page title for the section heading
        title_el = soup.find("h1") or soup.find("title")
        page_title = title_el.get_text(strip=True) if title_el else page["subtopic"]

        text = _extract_main_content(soup)
        if len(text) < 100:
            log.warning("    Too short (%d chars) — skipping", len(text))
            time.sleep(DELAY)
            continue

        doc = {
            "doc_id": f"mom_{page['topic'].lower().replace(' ', '_')}_{page['subtopic'].lower().replace(' ', '_')[:30]}",
            "source": "mom",
            "act_name": f"MOM Guidance — {page['topic']}",
            "section_id": "",
            "section_title": page_title,
            "topic": page["topic"],
            "subtopic": page["subtopic"],
            "url": page["url"],
            "date_retrieved": TODAY,
            "text": text,
        }
        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
        total += 1

        # Also download any PDFs linked from the page
        pdf_docs = scrape_linked_pdfs(
            soup, page["url"], SESSION,
            source="mom",
            act_name=f"MOM Guidance — {page['topic']}",
            today=TODAY,
            id_prefix=f"mom_{page['topic'].lower().replace(' ', '_')}",
        )
        for pdoc in pdf_docs:
            fout.write(json.dumps(pdoc, ensure_ascii=False) + "\n")
        total += len(pdf_docs)

        time.sleep(DELAY)
    return total


def _scrape_ask_gov(fout) -> int:
    """
    Scrape ask.gov.sg Q&A pairs for MOM.

    ask.gov.sg structure (as of 2025): the agency page lists questions as
    <div class="searchable-answer"> or similar containers. Each question has
    a title and an answer body. If the structure has changed, inspect the
    page source and update the selectors below.
    """
    total = 0
    for topic_url in ASK_GOV_TOPICS:
        log.info("  ask.gov.sg: %s", topic_url)
        soup = _get(topic_url)
        if soup is None:
            time.sleep(DELAY)
            continue

        # Try multiple candidate selectors for ask.gov.sg Q&A blocks
        qa_blocks = (
            soup.find_all("div", class_=lambda c: c and "answer" in c.lower())
            or soup.find_all("div", class_=lambda c: c and "question" in c.lower())
            or soup.find_all("article")
            or soup.find_all("li", class_=lambda c: c and "result" in (c or "").lower())
        )

        if not qa_blocks:
            # Fall back: save the whole page as one document
            log.warning("    No Q&A blocks found — saving full page")
            text = _extract_main_content(soup)
            if len(text) >= 100:
                doc = {
                    "doc_id": "ask_gov_mom_full",
                    "source": "mom",
                    "act_name": "MOM Q&A (ask.gov.sg)",
                    "section_id": "",
                    "section_title": "MOM Q&A Overview",
                    "url": topic_url,
                    "date_retrieved": TODAY,
                    "text": text,
                }
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total += 1
            time.sleep(DELAY)
            continue

        for idx, block in enumerate(qa_blocks):
            q_el = block.find(["h2", "h3", "h4", "strong", "dt"])
            question = q_el.get_text(strip=True) if q_el else ""
            answer = block.get_text(separator="\n", strip=True)
            if len(answer) < 50:
                continue
            text = f"Q: {question}\n\nA: {answer}" if question else answer
            doc = {
                "doc_id": f"ask_gov_mom_{idx}",
                "source": "mom",
                "act_name": "MOM Q&A (ask.gov.sg)",
                "section_id": str(idx),
                "section_title": question[:120] if question else f"Q&A {idx}",
                "url": topic_url,
                "date_retrieved": TODAY,
                "text": text,
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total += 1

        time.sleep(DELAY)

    return total


def scrape() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "mom.jsonl"
    total = 0

    with out_path.open("w", encoding="utf-8") as fout:
        total += _scrape_mom_pages(fout)
        total += _scrape_ask_gov(fout)

    log.info("MOM done: %d documents → %s", total, out_path)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scrape()
