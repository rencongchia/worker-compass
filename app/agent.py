"""
WorkerCompass agent pipeline.

Steps (in order):
  1. PII strip — remove FIN, NRIC, phone, email from query before logging or LLM call
  2. Language routing — language is user-selected in the UI; non-English triggers translation
  3. Translation — non-English → English via primary LLM (~50/30 tokens), fallback on 429
  4. Hybrid retrieval — BM25 + BGE-M3 → RRF → top-8 chunks
  5. Coverage check — if top-1 cosine similarity < threshold → refuse
  6a. REFUSE — return pre-translated refusal message in user's language
  6b. GENERATE — direct OpenAI-compatible API call, primary then fallback
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from openai import APIError, OpenAI, RateLimitError

from app.config import Config
from app.prompts import (
    CHUNK_TEMPLATE,
    DISCLAIMERS,
    FRESHNESS_WARNING,
    GENERATION_SYSTEM,
    GENERATION_USER,
    LANGUAGE_NAMES,
    REFUSALS,
    TRANSLATION_PROMPT,
)
from app.retrieval import HybridRetriever

log = logging.getLogger(__name__)

# Chunks below this dense cosine similarity are filtered from the LLM context
# to prevent BM25 false-positives (high keyword overlap, low semantic relevance)
# from confusing the generator. Always keep at least 3 chunks regardless.
_MIN_CHUNK_SIM_FOR_GEN = 0.28

# ---------------------------------------------------------------------------
# PII patterns — strip before any logging or LLM call
# ---------------------------------------------------------------------------
_PII_PATTERNS = [
    (re.compile(r"\b[STFG]\d{7}[A-Z]\b"), "[FIN/NRIC]"),           # Singapore FIN/NRIC
    (re.compile(r"\b\d{3}[-\s]?\d{4}[-\s]?\d{4}\b"), "[PHONE]"),    # phone numbers
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
]


def strip_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# Provider error markers — OpenRouter returns these as 200 OK text when a model is unavailable
_PROVIDER_ERROR_MARKERS = (
    "no endpoints found",
    "no such model",
    "model not found",
    "invalid model",
    "provider returned error",
    "this model's maximum context length",
)


def _is_provider_error_response(text: str) -> bool:
    t = text.strip().lower()
    return len(text) < 300 and any(m in t for m in _PROVIDER_ERROR_MARKERS)


def check_corpus_freshness(chunk_snapshot_dates: list[str], language: str = "en") -> str:
    today = date.today()
    oldest_days = 0
    for ds in chunk_snapshot_dates:
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
            oldest_days = max(oldest_days, (today - d).days)
        except ValueError:
            continue
    if oldest_days > 90:
        template = FRESHNESS_WARNING.get(language, FRESHNESS_WARNING["en"])
        return template.format(days=oldest_days)
    return ""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    answer: str
    language: str
    refused: bool
    top_similarity: float
    citations: list[dict] = field(default_factory=list)
    freshness_warning: str = ""


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class WorkerCompassAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.retriever = HybridRetriever(
            bm25_path=cfg.bm25_path,
            lancedb_path=cfg.lancedb_path,
            top_k=cfg.top_k,
            bm25_candidates=cfg.bm25_candidates,
            dense_candidates=cfg.dense_candidates,
        )
        self._primary = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
        self._fallback = (
            OpenAI(base_url=cfg.fallback_base_url, api_key=cfg.fallback_api_key)
            if cfg.fallback_api_key else None
        )

    def _llm_call(self, prompt: str, max_tokens: int = 100) -> str:
        """Lightweight single-turn LLM call (translation)."""
        def _call(client: OpenAI, model: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()

        use_fallback = False
        try:
            result = _call(self._primary, self.cfg.llm_model)
            if _is_provider_error_response(result) and self._fallback:
                log.warning("Primary LLM returned provider error; using fallback for translation")
                use_fallback = True
            else:
                return result
        except (RateLimitError, APIError) as exc:
            if self._fallback:
                log.warning("Primary LLM unavailable for translation (%s); using fallback", type(exc).__name__)
                use_fallback = True
            else:
                raise

        if use_fallback:
            return _call(self._fallback, self.cfg.fallback_model)

    def _translate_to_english(self, query: str) -> str:
        prompt = TRANSLATION_PROMPT.format(query=query)
        return self._llm_call(prompt, max_tokens=150)

    _SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    def _format_chunks(self, chunks: list[dict]) -> str:
        parts = []
        for idx, c in enumerate(chunks, 1):
            sup = str(idx).translate(self._SUP)
            parts.append(CHUNK_TEMPLATE.format(
                sup=sup,
                act_name=c.get("act_name", ""),
                section_title=c.get("section_title", ""),
                url=c.get("url", ""),
                text=c["text"][:800],
            ))
        return "\n\n---\n\n".join(parts)

    def run(self, query: str, language: str, history: list[dict] | None = None) -> AgentResponse:
        # 1. PII strip — clean before any LLM call or logging
        clean_query = strip_pii(query)
        lang_name = LANGUAGE_NAMES.get(language, "English")

        # 2 & 3. Translate to English if needed
        if language == "en":
            en_query = clean_query
        else:
            log.info("Translating from %s to English …", lang_name)
            try:
                en_query = self._translate_to_english(clean_query)
                log.info("Translated: %s", en_query[:120])
            except Exception as exc:
                log.error("Translation failed: %s — using original query", exc)
                en_query = clean_query

        # 4. Hybrid retrieval
        result = self.retriever.retrieve(en_query)
        log.info(
            "Retrieved %d chunks (bm25=%d, dense=%d, top_sim=%.3f)",
            len(result.chunks), result.bm25_hits, result.dense_hits, result.top_similarity,
        )

        # 5. Coverage check — refusal gate
        if result.top_similarity < self.cfg.similarity_threshold:
            log.info("Refusing: top_sim=%.3f < threshold=%.2f", result.top_similarity, self.cfg.similarity_threshold)
            return AgentResponse(
                answer=REFUSALS[language],
                language=language,
                refused=True,
                top_similarity=result.top_similarity,
            )

        # 5b. Corpus freshness check
        snapshot_dates = [c.get("corpus_snapshot_date", "") for c in result.chunks]
        freshness_warn = check_corpus_freshness(snapshot_dates, language)

        # 5c. Filter chunks passed to LLM — drop BM25 false-positives by dense similarity
        dense_filtered = [
            c for c in result.chunks
            if result.chunk_similarities.get(c["chunk_id"], 0.0) >= _MIN_CHUNK_SIM_FOR_GEN
        ]
        gen_chunks = dense_filtered if len(dense_filtered) >= 3 else result.chunks[:3]
        log.info("Chunks for generation: %d (filtered from %d)", len(gen_chunks), len(result.chunks))

        # 6. Generate
        history_block = ""
        if history:
            lines = []
            for turn in history[-2:]:
                lines.append(f"Worker: {turn['query']}")
                answer_preview = turn["answer"][:400]
                if len(turn["answer"]) > 400:
                    answer_preview += "…"
                lines.append(f"Assistant: {answer_preview}")
                lines.append("")
            history_block = "Previous conversation:\n" + "\n".join(lines) + "\n"

        system_msg = GENERATION_SYSTEM.format(language_name=lang_name)
        user_msg = GENERATION_USER.format(
            history_block=history_block,
            language_name=lang_name,
            original_query=clean_query,
            chunks=self._format_chunks(gen_chunks),
        )

        def _generate(client: OpenAI, model: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()

        use_fallback = False
        for attempt in range(2):
            try:
                client = self._fallback if use_fallback else self._primary
                model = self.cfg.fallback_model if use_fallback else self.cfg.llm_model
                answer = _generate(client, model)
                if attempt == 0 and self._fallback and _is_provider_error_response(answer):
                    log.warning("Primary LLM returned provider error %r; switching to fallback", answer[:80])
                    use_fallback = True
                    continue
                break
            except (RateLimitError, APIError) as exc:
                if attempt == 0 and self._fallback:
                    log.warning("Primary LLM unavailable (%s); switching to fallback", type(exc).__name__)
                    use_fallback = True
                else:
                    raise

        # Append disclaimer
        answer = answer.rstrip() + "\n\n" + DISCLAIMERS[language]

        # Build citations list — only chunks actually passed to the LLM
        citations = [
            {
                "act_name": c.get("act_name", ""),
                "section_title": c.get("section_title", ""),
                "url": c.get("url", ""),
                "snippet": c["text"][:300],
                "date_retrieved": c.get("date_retrieved", ""),
                "corpus_snapshot_date": c.get("corpus_snapshot_date", ""),
            }
            for c in gen_chunks
        ]

        return AgentResponse(
            answer=answer,
            language=language,
            refused=False,
            top_similarity=result.top_similarity,
            citations=citations,
            freshness_warning=freshness_warn,
        )
