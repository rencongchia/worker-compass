# worker-compass

## Link to demo video

https://drive.google.com/file/d/1T-SPlLhDLcvmDDkNiyxcbGHgGepY9mVm/view?usp=sharing

---

## What

WorkerCompass is a multilingual RAG chatbot that helps low-wage migrant workers in Singapore understand their employment rights. Workers can ask questions in **English, Bengali (বাংলা), Tamil (தமிழ்), or Burmese (မြန်မာ)** and receive grounded, cited answers drawn from Singapore legislation and official MOM guidance.

The system answers questions about salary disputes, work injury claims (WICA), employment agency fees, change-of-employer rights, repatriation, and the EFMA. It refuses out-of-scope questions and redirects workers to MOM, TWC2, or MWC rather than guessing.

---

## Why

MOM's employment rights guidance is published in English only. TADM mediation sessions frequently have no professional interpreter. Bangladeshi workers (~44% of dormitory-resident construction and marine process workers[^1]) speak Bengali. Indian workers (~40%[^1]) are often Tamil-speaking. Myanmar workers form another major source nationality in Singapore's construction sector.

These workers navigate salary disputes, work injury claims, and recruitment fee disputes without understanding the procedural options available to them. A language-aware legal information assistant directly addresses that information asymmetry.

The three gaps that justified building rather than reusing existing tools: no existing multilingual tool covers Bengali; no existing tool has a deterministic refusal gate; no existing tool cites specific legislation sections in the worker's own language.

### Current similar tools

| Tool | Scope | Languages | Key gap |
|---|---|---|---|
| **MigrantPal** (SMU/Reach Alliance, 2024)[^2] | General migrant worker support — insurance, employment, health, culture; Telegram chatbot | English + native languages via LLM translation | No RAG grounding; no refusal gate; LLM responses not source-cited; ~24% sustained adoption in dormitory trials |
| **Lawify** (Build for Good 2025) | Legal literacy — tenancy + employment claims; migrant workers, students, elderly | English only | No Bengali or Burmese; no retrieval-grounded citations |
| **Ask Jud** ({build} 2025) | Court procedures for self-represented litigants | English, Mandarin, Malay, Tamil | No Bengali or Burmese; courts-process scope, not employment rights |
| **Polyglot** (Hack for Public Good 2025) | Translation widget for Singapore government websites | Malay, Mandarin, Tamil | Translation only — no Q&A, no retrieval, no Bengali or Burmese |
| **Workers of Singapore** (Hack for Public Good 2024) | Migrant worker review portal, job matching, "red flags" education | English | Informational game, not a conversational RAG system |

WorkerCompass addresses three gaps no tool above closes simultaneously: Bengali and Burmese coverage for the two largest dormitory-resident nationality groups; a calibrated refusal gate that refuses out-of-corpus queries deterministically rather than hallucinating an answer; and source-grounded citations returned in the worker's own language. MigrantPal's own post-study write-up explicitly recommended adding RAG to reduce hallucination risk, which is the main focus of WorkerCompass.

---

## How

### Architecture

```
User query (EN / BN / TA / MY)
      │
      ▼
 PII strip (FIN, NRIC, phone, email redacted before any LLM call)
      │
      ▼
 Language routing (user-selected in UI)
      │  non-English?
      ▼
 Translation → English   (Groq LLM, ~50/30 tokens, ~200ms)
      │
      ▼
 Hybrid retrieval
   ├─ BM25 keyword search   (top-20 candidates)
   ├─ BGE-M3 dense search   (top-20 candidates, cosine similarity)
   └─ RRF fusion            (k=60) → top-8 chunks
      │
      ▼
 Refusal gate
   top-1 cosine similarity < 0.27 → refuse in worker's language
      │  passed
      ▼
 Per-chunk density filter
   drop chunks with sim < 0.28 from LLM context (BM25 false-positive guard)
      │
      ▼
 Corpus freshness check
   staleness warning appended if any source chunk > 90 days old
      │
      ▼
 Generation   (Groq Llama 3.3-70B → OpenRouter Qwen3 fallback on 429)
      │
      ▼
 Answer in worker's language + superscript citations + disclaimer
```

### Corpus

Scraped from MOM, TADM, and TAFEP using three per-source scrapers. Each HTML page is chunked by heading boundaries; linked PDFs are downloaded and split page-by-page. MOM's plain-English guidance pages cover the practical substance of WICA, EA Act, and Employment Act.

| Source | Content |
|---|---|
| MOM (mom.gov.sg) | Work permit conditions, salary rules, WICA, EA fee caps, repatriation |
| TADM (tal.sg) | Mediation process, ECT procedure, claim filing guides |
| TAFEP (tafep.sg) | Workplace Fairness Act, EFMA, fair hiring |

**505 chunks** across three sources (314 MOM, 116 TADM, 75 TAFEP). Indexes are committed to the repository for faster set-up. No scraping or embedding step required to run the app.

### Data Licensing and Privacy

**Source licensing.** All three sources are Singapore government websites. [MOM's Terms of Use](https://www.mom.gov.sg/terms-of-use) (clause 7) reserves all rights but permits fair dealing for private study, research, criticism, or review. Production deployment serving a public audience at scale would require written permission from MOM, TADM, and TAFEP respectively. The scrapers observe each site's `robots.txt`; all three sites allow automated crawling of their public guidance pages.

**Privacy.** The corpus contains no personal data, only published policy documents and guidance pages. User queries may contain personal information (FIN numbers, employer names, dispute amounts). The pipeline strips Singapore FINs/NRICs, phone numbers, and email addresses via regex before any LLM call or logging (`app/agent.py`, `strip_pii()`). Queries are sent to Groq (primary) or OpenRouter (fallback) API endpoints. A self-hosted Qwen3 deployment will eliminate this data-residency concern entirely.

### Embedding and Retrieval

**BGE-M3** (BAAI) for cross-lingual dense retrieval. Trained on [100+ languages](https://arxiv.org/abs/2402.03216); a Bengali query and its English translation produce similar vectors (avg. Recall@100 = 75.5 on [MKQA cross-lingual benchmark](https://arxiv.org/abs/2402.03216)). Runs locally via `sentence-transformers`; model weights are volume-mounted into the Docker container.

**Translate-then-retrieve** rather than direct cross-lingual retrieval: non-English queries are translated to English before retrieval so that BM25 can match Singapore-specific legal acronyms (WICA, TADM, ECT, In Lieu of Notice) that have no cross-lingual embedding alignment.

### LLM

**Primary:** Groq `llama-3.3-70b-versatile` (free, 30 RPM). Fast, reliable, no credit requirement.

**Fallback:** OpenRouter `qwen/qwen3-next-80b-a3b-instruct:free`, activated on 429 or provider errors from primary. Qwen3 is the preferred model for Bengali and Tamil. It [covers 119 languages](https://qwenlm.github.io/blog/qwen3/) including all three target languages, but its free-tier availability is less reliable than Groq.

**Production preference:** Self-hosted Qwen3 on a GPU instance. The codebase supports any OpenAI-compatible endpoint; switching requires changing two environment variables.

---

## Results / Evaluation of Current State

### Eval design

The evaluation suite (`eval/eval_set.json`) contains:
- **50 answerable English questions** across 6 categories: salary, WICA, EA fees, change of employer, repatriation, EFMA
- **10 out-of-corpus distractors** (unrelated topics that should be refused)
- **30 multilingual probes** (10 per language: Bengali, Tamil, Burmese)

Metrics:
| Metric | Method | Target |
|---|---|---|
| Refusal correctness | Binary: did the agent refuse all 10 distractors? | ≥ 90% |
| Citation accuracy | Deterministic: does answer mention expected act + section? | — |
| Answer correctness | LLM-as-judge (1–5), 95% CI via bootstrap | ≥ 4.0 |
| Answer groundedness | LLM-as-judge (1–5) — every claim traceable to retrieved chunk | ≥ 4.0 |
| Multilingual faithfulness | LLM-as-judge vs English gold answer (1–5) | — |

**Why this methodology.** Refusal correctness and citation accuracy are fully deterministic. No judge needed and no variance in judgement. LLM-as-judge was chosen for correctness, groundedness, and multilingual faithfulness because human legal evaluation at this scale requires domain expertise in Singapore employment law across four languages, which is not practically available. The judge is given the retrieved source chunks alongside the answer, so it can verify grounding without domain knowledge — reducing the dependency on legal expertise. The known limitation is self-grading bias: the same model (Groq Llama-3.3-70B) acts as both generator and judge. Majority voting across 3 judge runs (`--judge-runs 3`) reduces variance but does not eliminate this bias. Bootstrap 95% CI over 1,000 resamples is reported to surface the instability from small sample sizes. A production eval would use a held-out model (e.g. GPT-4o) as judge and include a native-speaker spot-check layer for the multilingual probes.

### Current status

Full eval is pending due to token budget constraints on the free-tier LLM (100k tokens/day on Groq). The `--quick` smoke test (~26 questions, ~70k tokens at 1 judge run) is the recommended first run. If you have your own API key, you can run the evaluations with it as well, as long as it is OpenAI-compatible endpoint.

To run the evaluation:

```bash
uv run python eval/run_eval.py --quick # ~26 questions
uv run python eval/run_eval.py --category salary # single category
uv run python eval/run_eval.py --quick --judge-runs 3 # majority-vote judge (>100k tokens)
uv run python eval/run_eval.py # full 90 questions
# Results written to eval/results/eval_<date>.json
```

---

## Deployment Considerations

**Who would run it and where.** The natural operators are MOM, TWC2, or MWC — agencies already providing employment rights support to migrant workers. A government deployment would sit on the Singapore Government Commercial Cloud (GCC), likely as a containerised Streamlit app behind a reverse proxy. A non-profit deployment could run on a single low-cost cloud VM (4 CPU, 8 GB RAM is sufficient for CPU-only BGE-M3 inference and Streamlit).

**Compute footprint at realistic scale.** Groq's free tier (30 RPM, ~100k tokens/day) handles roughly 50–100 questions per day before hitting ceilings. At 500 daily active users, a paid Groq subscription or self-hosted Qwen3-32B on a single A10G GPU instance (~$0.30/hour on Lambda Labs) is the realistic infrastructure. BGE-M3 CPU inference runs at ~200ms per query; a single instance handles ~5 concurrent users before latency degrades. Horizontal scaling requires a shared vector store (e.g. move LanceDB to object storage or switch to a hosted vector DB).

**Swapping the generator.** The system uses any OpenAI-compatible endpoint — switching providers requires only two environment variables (`LLM_BASE_URL` and `LLM_MODEL`). For a government deployment with data-residency requirements, this means routing to a self-hosted Qwen3-32B on a local GPU node with no query data leaving the network. For a higher-budget deployment, the same two variables can point to Claude or GPT-4o. AI Singapore's SEA-LION is the preferred generator for Tamil and Burmese in a production Southeast Asian deployment; it can be added as a second fallback once Bengali coverage is addressed.

**What to monitor once live.** Refusal rate per language (a spike signals corpus staleness or a new query type the corpus doesn't cover); LLM API error rate and fallback trigger frequency; median response latency; and thumbs-down feedback ratio per session. A persistent ≥ 20% thumbs-down rate warrants a prompt review.

**The risk that keeps me up at night.** Corpus staleness with high user confidence. A worker who receives a confidently cited but outdated answer about a filing deadline or compensation ceiling and acts on it may miss a claim window entirely. The 90-day freshness warning is a mitigation, not a solution. A production deployment needs a weekly automated ETL pipeline and a clear "last updated" timestamp visible to the user on every answer.

---

## Project set-up and running/deployment instructions

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- A [Groq](https://console.groq.com/) API key (free, no credit card required)

---

The corpus, BM25 index, and LanceDB vectors are committed to the repository — no scraping
or embedding step required. BGE-M3 weights are volume-mounted from a local
`models/` directory.

```bash
# 1. Clone the repository
git clone https://github.com/rencongchia/worker-compass.git
cd worker-compass

# 2. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Copy and fill in your API key
cp .env.example .env
# Edit .env: set GROQ_API_KEY

# 4. (Recommended) Download BGE-M3 model weights
#    uv will install project dependencies automatically on first run
uv run huggingface-cli download BAAI/bge-m3 --local-dir ./models
#    Skipping this means the container downloads ~2.2 GB on the first query (5–10 min).
#    Weights persist in ./models/ via volume mount — no re-download on subsequent starts.

# 5. Start Docker Desktop, then build and start the container
docker compose up --build
# → Open http://localhost:8501

# Subsequent starts (image already built)
docker compose up
```

> **If you skip step 4:** The app starts immediately at localhost:8501.
> BGE-M3 weights are lazy-loaded on the **first query** — if `models/` is empty the
> container downloads the model from HuggingFace at that point (5–10 min depending on
> your connection). The weights persist in `./models/` via the volume mount, so
> subsequent restarts load from disk with no re-download.

> **Why the corpus is pre-built:** The ETL pipeline scrapes live government websites and the
> embedding step takes ~10 minutes on CPU. In production, indexes would live in object
> storage and be rebuilt by a scheduled pipeline; see PROCESS.md Log 2 for details.

### Environment variables

Only one variable is required. Everything else has a working default.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Primary LLM (Llama 3.3-70B, free, 30 RPM) |
| `OPENROUTER_API_KEY` | No | — | Fallback LLM, activated on Groq rate-limit errors |
| `LLM_MODEL` | No | `llama-3.3-70b-versatile` | Override primary model |
| `LLM_BASE_URL` | No | `https://api.groq.com/openai/v1` | Override primary endpoint |
| `FALLBACK_MODEL` | No | `qwen/qwen3-next-80b-a3b-instruct:free` | Override fallback model |
| `FALLBACK_BASE_URL` | No | `https://openrouter.ai/api/v1` | Override fallback endpoint |
| `SIMILARITY_THRESHOLD` | No | `0.27` | Refusal gate cosine threshold (BGE-M3 range) |

---

## Limitations and future works

- **Bengali and Burmese retrieval benchmarking.** BGE-M3's published MKQA cross-lingual benchmark covers 25 languages but excludes Bengali and Burmese. The model embeds these languages, they are in its training data, but retrieval quality on them has not been formally measured. This is the largest unvalidated assumption in the system.
- **No automatic language detection.** If a user types Bengali but has the UI set to English, the query goes to retrieval untranslated and the answer is returned in English. A `langdetect` pre-pass would fix this but adds latency and a dependency.
- **Token budget for eval.** The free-tier Groq ceiling (100k tokens/day) makes running the full 90-question eval expensive. The `--quick` mode (~70k tokens) is the practical ceiling for daily runs. Formal evaluation at submission quality requires a paid API tier or spreading runs across days.
- **SSO statutory text missing from corpus.** Singapore Statutes Online is blocked by Cloudflare. MOM's plain-English guidance covers the practical substance, but the actual statutory text is not indexed. A production deployment would use a headless browser or a pre-downloaded PDF drop-in for each Act.
- **No native speaker validation.** Multilingual outputs were spot-checked but not reviewed by native-speaking legal aid workers. The refusal and disclaimer strings were machine-translated by Qwen3 and are explicitly flagged for native speaker review before production use.
- **WhatsApp / Telegram interface.** Dormitory-resident workers primarily communicate via messaging apps. A Twilio-based bot would reach significantly more users but adds infrastructure complexity outside the prototype scope.
- **SEA-LION for Tamil/Burmese in production.** AI Singapore's SEA-LION v3 (Gemma-SEA-LION) has strong Tamil and Burmese coverage and is the preferred generator for those languages in a self-hosted production deployment. It was not used here because Bengali is absent from its language set, and Groq does not host it. Routing Bengali queries to Qwen3 and Tamil/Burmese queries to SEA-LION would be an alternative architecture.

---

[^1]: Figures from a 2024 cross-sectional health survey of 1,001 male migrant workers in Singapore (95% dormitory-resident, 60% construction sector). Official MOM statistics do not publish a nationality breakdown of the foreign workforce. Source: [Building a healthy migrant workforce in Singapore](https://www.sciencedirect.com/science/article/pii/S2666623524000552), ScienceDirect, 2024.

[^2]: Chua, X., Tan, D., Oi, J., Tan, Y., & Koh, A. (2024). [Project MigrantPal: Harnessing Digital Technology to Improve the Well-being of Migrant Workers in Singapore](https://reachalliance.org). Reach Alliance / Singapore Management University. The study tested a Telegram-based chatbot across dormitory focus groups; the team's own conclusion was that RAG was needed to reduce hallucination and that Bengali support was the most-requested missing feature.
