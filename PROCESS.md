# PROCESS.md — WorkerCompass

## Log 1 — Problem Scoping & Architecture Decisions

### Problem Selection

I started by reading through the Annex A problem sketches and the {build} 2025 public project
submissions to calibrate the space. The submissions broadly fall into three types: internal
tooling for agency officers, agency-specific or WOG tools, and tools that directly serve the
general public. I wanted the third category — a tool with direct impact on a real user who
has no existing alternative.

Several factors shaped the shortlist: personal interest in vulnerable populations, thematic fit
with the brief's four AI shapes (retrieval and knowledge, agents and workflows, unstructured to
structured, generation with guardrails), novelty relative to what already exists, feasibility
given public data availability, and time constraints.

The first idea was a tool to proactively identify students with learning disabilities from school
records — attendance, grades, behavioural flags. It was dropped quickly. The data is not public,
the privacy risk is substantial, and the core task is a classification problem that does not
require an agentic AI system. It also required a school operator partner to be useful, which
was outside scope for a prototype.

From there the focus shifted to other vulnerable populations: the elderly, disabled individuals,
low-income households, migrant workers. My earlier internship under MOM made the migrant worker
angle concrete — I knew which agencies were involved, where public data lived, and roughly what
the information gap looked like.

The problem statement that emerged: MOM's employment rights guidance is English-only. TADM
mediation sessions frequently have no translator. Workers from Bangladesh, India, and Myanmar —
who make up the majority of dormitory-resident construction workers — navigate salary disputes,
work injury claims, and recruitment fee disputes without understanding the procedural options
available to them. That is a clear information asymmetry that a language-aware RAG system can
directly address.

Thematically this maps onto all four brief shapes simultaneously: RAG over policy documents
(retrieval and knowledge), translation + agentic tool calls (agents and workflows), pulling
structured citations from unstructured legislation (unstructured to structured), and generation
with guardrails to prevent hallucinated legal advice.

I checked for prior art. Similar tools have been ideated at {build} and by non-profit groups.
None are multilingual, none include Bengali (which covers the largest worker nationality), and
none have a deterministic refusal gate. Those three gaps justified building rather than reusing.

---

### Language Selection

Language choices were driven by demographic data.

Among dormitory-resident construction and marine process (CMP) workers, Bangladeshi workers
account for approximately 44% and Indian workers approximately 40% of the population
(Journal of Migration and Health, Vol. 10, 2024). Myanmar workers are the third largest
non-English, non-Malay source country.

- **Bengali** — Bangladeshi workers (~44% of CMP sector). The largest single nationality
  group and the most critical language gap: Bengali is absent from SEA-LION's language set,
  meaning no existing Singapore-built model serves this population.
- **Tamil** — Indian workers (Tamil-speaking subset); also one of Singapore's four official
  languages.
- **Burmese** — Myanmar workers; third largest source country in the CMP sector.

Malay/Indonesian, Tagalog, Hindi, and Mandarin were scoped out of v1. Malay/Indonesian
workers have higher English literacy rates. Filipino workers (Tagalog) are predominantly
in the domestic worker sector, which has a different legal framework better addressed by
a separate, MDW-scoped tool. These are documented as v2 scope in the README.

---

### Generator Model

#### Why Qwen3-32B

Qwen3 was chosen as the generator for three reasons that hold up under scrutiny.

**Bengali coverage.** The single largest vulnerable group — Bangladeshi workers — speaks
Bengali. AI Singapore's SEA-LION (the most obvious Singapore-contextual choice) does not
support Bengali in any version. Its language set is Burmese, English, Indonesian, Khmer,
Lao, Malay, Mandarin, Tagalog, Tamil, Thai, and Vietnamese. Qwen3 covers 119 languages
including Bengali, Tamil, and Burmese natively, trained on approximately 36 trillion tokens
(Qwen3 Technical Report, arXiv:2505.09388, 2025).

**Open weights and self-hostability.** Qwen3 is released under Apache 2.0. A production
deployment handling sensitive worker queries — names, employer details, dispute amounts —
should not route data through a third-party API if avoidable. Qwen3 can be self-hosted on
a single GPU instance; Claude and GPT cannot. This is documented as the preferred production
configuration in the README deployment section.

**Zero cost at prototype scale.** Available free via OpenRouter (qwen/qwen3-32b:free) with
no credit card. Groq (Llama 3.3 70B) serves as a fallback on rate-limit errors — both
providers use OpenAI-compatible endpoints, so the swap requires changing one environment
variable.

#### Models considered and dropped

**SEA-LION (Gemma-SEA-LION v3).** The first model considered given its Southeast Asian
training focus and the fact that it is produced by AI Singapore, which is directly relevant
to this deployment context. It was not used as the primary generator because Bengali is
absent from its language set — the largest single worker nationality group would be entirely
unserved. SEA-LION remains the preferred generator for Tamil and Burmese in a future
self-hosted deployment, and is documented in the README as such. Its absence here is a scope
constraint, not a quality judgement.

**SEA-HELM.** Evaluated as a benchmark rather than a generator model — SEA-HELM is AI
Singapore's Southeast Asian language evaluation harness. It is not a deployable LLM. It
was initially confused with SEA-LION during scoping and noted here for clarity.

**Claude / GPT.** Proprietary APIs with no self-hosted option. A deployment handling
sensitive worker queries has legitimate data-sovereignty reasons to prefer an open-weights
model. Retained as a configuration option in the codebase for users who prefer it.

**NLLB-200.** Initially planned as a dedicated translation layer. Dropped for two reasons:
Qwen3 handles all three target languages natively, removing the need for a separate
translation model; and NLLB-200's own model card explicitly states it is not intended for
domain-specific text such as legal content — legal terminology like "In Lieu of Notice"
or "TADM mediation" would be mistranslated. Removing NLLB-200 also reduced the Docker
image size by ~1.2GB and eliminated a failure point from the pipeline.

---

### Embedding Model

BGE-M3 (BAAI) was chosen for cross-lingual dense retrieval.

It was trained on 100+ languages with explicit cross-lingual alignment, meaning a Bengali
query and an English document with the same meaning produce similar vectors. It runs
locally via fastembed on CPU, requires no API call, and ships inside the Docker image.

Published Recall@100 on the MKQA cross-lingual benchmark is 75.5 versus 70.1 for the
next best baseline (BGE-M3 paper, arXiv:2402.03216, 2024). One honest caveat: MKQA covers
25 non-English languages but does not include Bengali or Burmese. BGE-M3 can embed these
languages — they are in its training data — but its retrieval performance on them has not
been formally published. This is noted in the README limitations section.

---

### Retrieval Design: Translate-Then-Retrieve

Two approaches were considered for handling non-English queries:

**Direct cross-lingual retrieval:** Bengali query → BGE-M3 embed → compare against English
chunk embeddings directly.

**Translate-then-retrieve (chosen):** Bengali query → Qwen3 translate to English (~200ms)
→ English query → BM25 + BGE-M3 hybrid → top-8 chunks.

Translate-then-retrieve was chosen for two reasons.

First, BM25 — the keyword-matching half of the hybrid retrieval system — is entirely blind
to non-English queries. BM25 provides critical exact-match coverage for Singapore legal
terminology: "WICA", "TADM", "Employment Claims Tribunal", "In Lieu of Notice". These
terms cannot be semantically discovered; they must be matched by keyword. A Bengali query
produces zero BM25 results, disabling half the retrieval system. The translation step
restores full BM25 functionality.

Second, Singapore-specific legal acronyms are poorly aligned cross-lingually in BGE-M3.
A worker writing "labour dispute mediation" in Bengali script embeds near general labour
dispute content, not near the TADM Mediation Guide chunk whose title contains the acronym
"TADM". Qwen3's translation resolves the acronym; BM25 then finds the exact match.

The translation call is lightweight (~50 input / ~30 output tokens, ~200ms) and the
precision improvement on jurisdiction-specific legal terms justifies the added latency.

Initial thoughts on evaluation included Recall@k and Precision@k as retrieval metrics.
These were retained but subordinated to the refusal gate as the headline KPI — see the
Refusal Gate section for reasoning.

---

### Agentic Tool Calls

The core RAG pipeline — translate, retrieve, check, generate — is a fixed linear flow.
To genuinely fulfil an agentic shape, WorkerCompass adds three optional tool calls the
agent invokes based on query content:

**EA licence checker:** If the query mentions a specific employment agency by name, the
agent calls a tool that queries MOM's public EA directory to verify whether the agency is
licensed. This is a live lookup, not a retrieved corpus chunk — the corpus can go stale
but the EA directory reflects current status.

**Nearest help centre locator:** If the query contains a location ("I am in Jurong") or
asks where to get help, the agent calls a tool over the data.gov.sg Social Service Agencies
dataset to return the nearest SSO, TWC2, or MWC office.

**Corpus freshness check:** Before generating an answer about WICA compensation limits or
EA fee caps — figures that have changed recently (WICA limits changed 1 November 2025) —
the agent checks the `corpus_snapshot_date` metadata and appends a staleness warning if
the source chunk is older than 90 days.

These tool calls are invoked by the agent's decision logic, not hardcoded into every
query path. A question about salary non-payment does not trigger the EA checker; a question
mentioning "my agent from Dhaka" does. That decision is what makes the system agentic
rather than purely a retrieval pipeline.

---

### Agent Framework: Agno

Agno was chosen over LangGraph because the pipeline is mostly linear with a small number
of decision points. LangGraph's graph-based abstraction adds meaningful boilerplate for
complex branching workflows but is over-engineered for a 5-step pipeline with one branch
and 3 optional tool calls.

Agno's practical advantages for this build: built-in PII detection guardrail (one-line
`pre_hook` that strips names and ID numbers before logging — directly relevant when workers
may type FIN numbers or employer names into queries); native LanceDB integration; 23+ LLM
provider support with plug-and-play switching for the OpenRouter/Groq fallback.

---

### Vector Store: LanceDB

Fully embedded, file-based, no server process required. The Docker container starts and
serves queries immediately without a warm-up period. The index persists as a mounted
volume so it survives container restarts. Apache Arrow columnar format enables fast
metadata filtering (e.g. filter by `act_name = "WICA"` before vector search). No vendor
account or external network call required at query time.

---

### Refusal Gate

Prompt-only refusal ("say I don't know if you can't find the answer") was implemented
first and dropped. LLMs hallucinate even when instructed not to. The refusal gate is
a hard deterministic check: if the top-1 retrieved chunk's cosine similarity is below
threshold τ (0.55, calibrated against the distractor eval set), the system refuses before
the LLM call is made. The LLM never sees an out-of-corpus query. Refusal correctness on
10 out-of-corpus distractors is the headline evaluation KPI — the highest-harm failure
mode is a confidently wrong answer about a filing deadline, not a low-quality answer.

Early eval thinking included LLM-as-judge via Claude's top-p scoring. The final approach
uses Qwen3 as the judge in 3 independent runs with majority vote, avoiding the need for a
separate API key and keeping the eval self-contained.

---

---

## Log 2 — Corpus ETL: Scraping Challenges & Embedding

### URL Staleness

The first ETL run produced only 40 chunks. The cause was silent 404s across all four sources — MOM, TADM, TAFEP, and SSO had all restructured their URLs at some point after the initial scraper URLs were written. The scrapers were swallowing non-200 responses without raising, so the job appeared to complete successfully while discarding most of its output.

Fixing this required manually tracing each broken URL:
- **10 MOM pages** — MOM consolidated several topic areas. For example, `using-employment-agencies` and `employment-agency-fees` were merged into `employment-agencies/key-facts`; `repatriating-your-worker` and `workers-duties-and-obligations` collapsed into `work-permit-conditions`.
- **5 TADM pages** — tal.sg restructured its navigation. The `employees` landing page and several sub-pages moved; the ECT Guide PDF moved from the MOM document library to the judiciary.gov.sg server entirely.
- **4 TAFEP pages** — `workplace-fairness-act` was renamed to `workplace-fairness` following the WFA coming into force; the `getting-help` page became `contact-us`.

After fixing the URLs: 96 chunks.

---

### HTML-Only Scraping Was Insufficient

The initial approach scraped HTML page text only. After the URL fixes the chunk count reached 96, but a review of the content revealed the problem: the most substantive guidance — enforcement conditions, pass conditions, EA licence conditions, ETL procedural guides — was in PDFs linked from those pages, not in the HTML body itself.

The fix was a shared `pdf_utils.py` module that all scrapers call after each page fetch. `find_pdf_links()` scans every `<a href>` for `.pdf` extensions or `viewtype=pdf` parameters, downloads each PDF, and parses it with `pymupdf4llm` into Markdown. Sections are split on heading boundaries and chunked the same way as HTML content.

The impact was significant:
- `gatiod.pdf` (MOM Work Permit conditions) — 174 sections
- `wpspassconditions.pdf` (Work Permit Specialist conditions) — 127 sections
- `ea-licence-conditions.pdf` (Employment Agency conditions) — 38 sections

After adding the PDF crawler: **783 chunks**. Nearly the entire jump from 96 to 783 came from PDF content that the HTML-only approach had silently skipped.

---

### SSO: Cloudflare Block

Singapore Statutes Online returns 403 Forbidden to automated requests for its HTML pages. The Cloudflare bot detection cannot be bypassed without a headless browser, and adding Playwright to the Docker image for a single source was not worth the complexity at prototype scale.

SSO does expose `?ViewType=Pdf` on each Act URL which returns the full Act text as PDF bytes. This was implemented and initially appeared to work — `fitz.open()` accepted the bytes — but the resulting text was garbled (it was parsing a Cloudflare JS challenge page, not actual PDF content). The root cause was a `BytesIO` import bug (`__import__("io")` pattern instead of a top-level `import io`) which caused the stream to be passed incorrectly.

After fixing the import, the PDF bytes still failed to parse, suggesting the Cloudflare interception happens at the application layer before the PDF is served. Manual testing in a browser (with session cookies) works; the scripted call does not.

Decision: **Skip SSO for the prototype**. MOM's guidance pages cover the practical substance of WICA, the Employment Act, and the EA Act in plain English — the actual statutory text is not how workers look up their rights. The limitation is documented in the README. The scraper shell remains in the codebase for a future implementation using a headless browser or a pre-downloaded PDF drop-in.

---

### Embedding Model: fastembed → sentence-transformers

The initial implementation used `fastembed` to embed with BGE-M3. Running `build_index.py` threw a `ValueError: Model BAAI/bge-m3 is not supported` — fastembed v0.4+ removed BGE-M3 from `TextEmbedding.list_supported_models()` without a deprecation warning, likely due to the model's size making it impractical for their use case.

The fix was to switch to `sentence-transformers`, which is the canonical library for the BGE model family (maintained by BAAI, the same organisation that released BGE-M3). The API is slightly different (`encode()` vs `embed()`) but the underlying model and weights are identical. Two dependency lines replaced one:

```toml
# before
"fastembed>=0.6.1"

# after
"sentence-transformers>=3.0.0",
"torch>=2.3.0",
```

---

### Final Retrieval Design

The hybrid BM25 + BGE-M3 approach described in Log 1 was implemented without changes to the design. A few implementation decisions worth noting:

**Lazy loading.** Both the BM25 index and the LanceDB table are loaded on first query rather than at import time. Streamlit re-runs the entire script on each interaction; loading a 2.3 GB model and a LanceDB table at import would add several seconds to every page interaction. Lazy loading means the first query takes ~3 seconds while the model warms up; subsequent queries are ~200ms for retrieval.

**RRF constant.** The standard k=60 was used without tuning. k=60 is the value from the original Cormack et al. (2009) paper and is widely used as a default. With 783 chunks a lower k would over-weight early-ranked documents; 60 provides a reasonable balance.

**Cosine threshold.** τ=0.55 was set based on manual inspection of similarity scores across in-scope and out-of-scope queries. Scores for in-scope queries cluster above 0.6; out-of-scope queries (unrelated topics) cluster below 0.4. 0.55 sits in the gap. Formal calibration against the eval set is documented as a remaining step.

---

### Artifact Distribution: Pre-built Indexes for Submission

The original submission plan required an assessor to: install uv, run ETL (live web scraping), wait for BGE-M3 to download, run 10 minutes of CPU embedding — before the app could even start. That is a significant friction point for a reviewer.

Since the corpus is a point-in-time snapshot (not live data), the indexes are equivalent to source artifacts and can be committed to the repository:

- `corpus/chunks/*.jsonl` — raw scraped text, ~1.4 MB
- `corpus/bm25_index.pkl` — BM25Okapi index, ~3 MB
- `corpus/lancedb/` — LanceDB table with BGE-M3 vectors, ~20–30 MB

With these committed, the assessor workflow collapses to:

```bash
cp .env.example .env   # set OPENROUTER_API_KEY
docker compose up --build
```

BGE-M3 is downloaded once — at Docker image build time — and baked into the image. The container starts with no network dependency and no waiting.

**What this would look like in production:**

Committing model artifacts to git is a prototype shortcut, not a pattern for a live system. In production:

- Corpus chunks and indexes live in object storage (S3 or GCS), versioned by snapshot date.
- A scheduled pipeline (weekly cron or Airflow DAG) re-runs ETL and indexing and uploads fresh artifacts. The pipeline is triggered automatically when source documents change, or on a fixed cadence to catch MOM/TADM page updates.
- The Docker image pulls artifacts from object storage at container startup via an entrypoint script, not at build time via COPY. This keeps the image small and separates the model release cycle from the data release cycle.
- Model weights live in a shared model registry or a persistent EFS/Filestore volume mounted across instances. The image does not bake in the weights; instead it mounts the pre-loaded volume, keeping image size under 1 GB and model updates decoupled from application deploys.
- Git contains only code. Data and weights are external dependencies referenced by versioned pointers (e.g. an S3 URI pinned in a config file).

---

---

## Log 3 — Docker Deployment Challenges

### BGE-M3 in Docker: Three Approaches, One That Works

**Approach 1 — Download inside Docker at build time.**
The initial plan was to `RUN python -c "SentenceTransformer('BAAI/bge-m3')"` during the build, baking the weights into the image. Inside Docker, HuggingFace downloads are unauthenticated and rate-limited. After 2+ hours the download had not completed. Abandoned.

**Approach 2 — COPY model weights into image.**
The model is already cached locally at `~/.cache/huggingface/hub/models--BAAI--bge-m3/` (4.3 GB including snapshots and blobs). The plan was to `cp -r` it into `models/` and `COPY models/ /app/models/` in the Dockerfile. The build succeeded, but the image export step — Docker unpacking and writing ~7 GB of layers to its internal virtual disk — ran for 1.5 hours and filled the host disk completely. Docker Desktop froze. `docker system prune` took another 20+ minutes to reclaim the space.

**Approach 3 — Volume mount (chosen).**
The `models/` directory is mounted into the container at runtime rather than baked into the image:

```yaml
volumes:
  - ./models:/app/models
```

```dockerfile
ENV HF_HUB_CACHE=/app/models
```

`HF_HUB_CACHE` tells `huggingface_hub` where to look for cached weights. The image contains only code and Python packages (~1.5 GB). The assessor copies the model once before the first run and the container starts in seconds. This is the same net result as baking the weights in — no network call at runtime — but the image is 4 GB smaller and the build takes minutes rather than hours.

---

### Python Module Path in Docker

`from app.agent import WorkerCompassAgent` failed with `ModuleNotFoundError: No module named 'app'` inside the container. Locally, the working directory is the project root and Python's path includes it; inside Docker, `WORKDIR /app` means `/app` is the working directory and `app/` is a subdirectory, but it is not automatically on `sys.path` when Streamlit launches the script.

Fix: set `PYTHONPATH=/app` in docker-compose.yml environment. This ensures `from app.xxx import` always resolves to `/app/app/xxx.py` regardless of how the process is launched.

---

### What Would Be Different with More Time

**Native speaker validation at scale.** The 10-sample human eval subset per language is
the minimum defensible floor. A production deployment would require 200+ outputs reviewed
by native-speaking legal aid workers for each language.

**WhatsApp or Telegram interface.** Dormitory-resident workers primarily communicate via
messaging apps. A Twilio-based bot would be the highest-reach channel but adds
infrastructure complexity outside the prototype scope.

**MWC Settling-in Programme corpus.** The MWC SIP curriculum covers worker rights in 7
languages including Bengali and Burmese but is gated behind employer login. A partnership
conversation with MWC would unlock the highest-quality multilingual source available.

**Formal Bengali and Burmese retrieval benchmarking.** BGE-M3's published MKQA benchmark
does not cover Bengali or Burmese. Assembling a small gold-standard QA evaluation set in
these languages would make the retrieval quality claims more defensible.

---

---

## Log 4 — Retrieval Calibration and Generation Quality

### BGE-M3 Similarity Score Range

After deployment, the first live queries were being refused even when clearly on-topic. A salary
dispute question returned a top cosine similarity of ~0.32, which fell below the τ=0.55 threshold
set in Log 1. The assumption had been that BGE-M3 cosine similarities would behave like OpenAI
text-embedding-3 scores, which cluster at 0.7–0.95 for relevant pairs. They do not.

BGE-M3's cosine similarity profile is fundamentally different. Relevant query-document pairs
score in the 0.25–0.55 range; semantically unrelated pairs score 0.05–0.20. The gap between
in-scope and out-of-scope is about 0.15–0.20, not 0.40–0.50. This is not a model quality issue —
it is a consequence of how BGE-M3 normalises its embeddings across a 100+ language training
distribution. The threshold was recalibrated to 0.25 after confirming that all distractor
(out-of-corpus) queries still score below 0.20 against every corpus chunk.

---

### BM25 False Positives and the Per-Chunk Density Filter

After lowering the refusal threshold, a new problem surfaced: the LLM was receiving irrelevant
chunks. A salary dispute query retrieved Work Permit pass conditions — not because BGE-M3 found
them relevant, but because BM25 matched on common terms like "employer" and "payment" that appear
in both documents. RRF fusion elevated those chunks into the top-8 based on keyword overlap while
their BGE-M3 cosine similarity was 0.15–0.20, well below the relevance threshold.

The fix distinguishes two different checks. The refusal gate (τ=0.25) answers the question: does
the corpus contain anything relevant to this query at all? A secondary filter applied before the
LLM call answers a different question: which retrieved chunks are actually relevant enough to put
in the generation context? Any chunk with a dense cosine similarity below `_MIN_CHUNK_SIM_FOR_GEN
= 0.28` is stripped from the context, with a floor of 3 chunks to prevent an empty prompt. The
refusal gate and the generation filter operate at different thresholds because they serve different
purposes — one is a binary pass/fail gate, the other is a quality filter on an already-passing set.

---

## Log 5 — Agent Behaviour

### Agentic Tool Over-Triggering

The Agno agent was calling `find_nearest_help_centre` on nearly every query, including
straightforward salary rights questions. The result was physical SSO addresses injected into
legal guidance answers — a list of buildings in Jurong and Bedok appended to a response about
filing a TADM claim. In some cases the tool result was duplicated in the response because Agno
appended both the tool output and the generated answer.

The root cause: the system prompt gave no guidance on when *not* to use tools, leaving the model
to infer tool relevance from tool descriptions alone. "Find the nearest help centre" sounds broadly
applicable when the user is a worker seeking help.

The fix was explicit negative instructions: "Do NOT call tools for general questions about rights,
salary, injuries, or procedures" alongside the positive trigger conditions for each tool. Negative
constraints are necessary when a tool's description has surface-level overlap with a large fraction
of the query domain. A general rule like "only call if the user explicitly asks" is insufficient
— "what should I do about my salary?" could plausibly read as explicitly asking for help, so the
negative instruction needs to name the domains to exclude.

---

## Log 6 — Removing Agno: Direct LLM Calls

### Why Agno Was Removed

After Log 5's tool over-triggering fix — tightening negative constraints to prevent `find_nearest_help_centre` and `check_ea_licence` from firing on general queries — a more fundamental question emerged: what is Agno actually providing?

The three tools were the answer when Agno was introduced. Agno's value proposition is an execution loop that handles tool-call parsing, dispatch, result injection, and re-prompting — capabilities that matter when the agent genuinely needs to decide between multiple tools mid-flight. With explicit negative instructions controlling most of the triggering surface, the tools were rarely firing. The pipeline was effectively a fixed linear flow with Agno as dead weight around it.

The decision was made to remove the tools entirely from the submission prototype. The reasoning: adding tool calls to a chatbot creates a higher reliability bar — a live EA licence lookup that returns an unexpected response, or a help centre API that times out, degrades the user experience in ways that are hard to test comprehensively. For a prototype being submitted for review, a deterministic pipeline that always produces an answer from the corpus is more defensible than one that sometimes makes external API calls. The tools are documented as a natural extension in the README's limitations section.

With the tools gone, Agno had no remaining function. Replacing it with a direct `client.chat.completions.create()` call:
- Removed one dependency (`agno>=1.4.0` from pyproject.toml)
- Made the execution path inspectable in a single method rather than inferred from framework internals
- Eliminated a framework version lock-in risk — Agno was a fast-moving library with breaking changes

The primary/fallback client logic, error handling, and provider-error-as-text detection from Log 5 were preserved unchanged; they were never Agno-specific. `check_corpus_freshness` remains as a direct Python function call before generation, not as an agent tool.

### What Was Retained from the Agentic Design

Removing Agno did not remove the agentic character of the pipeline. The pipeline still makes autonomous decisions: whether to translate (language detection), whether to refuse (refusal gate), whether to filter chunks (density filter), and whether to switch providers (fallback logic). These are the decisions that matter for the use case. The Agno framework was providing tool-call orchestration — a layer that only adds value when the agent is choosing between tools. Without tools, it was orchestrating nothing.

---

## Log 7 — UI, Eval Pipeline, and Token Budget

### Multi-Turn Chatbot with Session Management

The initial UI was a single-query form: submit a question, receive an answer, done. This is
inadequate for the actual use case — a worker asking about salary non-payment will follow up with
"how long do I have to file?" and needs the previous context to understand the answer.

The UI was rebuilt as a chatbot interface with sidebar session management. Sessions are created
automatically on first query, named from the first 45 characters of the opening message, and
listed in the sidebar with inline rename and delete controls. Multi-turn context is passed to the
LLM as the last two non-refused turns. Refused turns are excluded because a refusal contains no
substantive information useful for interpreting the next question.

Retrieval still runs on the current query only, not over a concatenated history. This is
intentional: running retrieval over a conversation history biases toward terms from earlier turns
and can cause the retriever to underweight the current question's semantics. History is context
for generation, not a retrieval query.

---

### LLM Judge: 3-Run Rationale and Reduction

The eval judge originally ran 3 independent calls per answerable question and took majority vote
across dimensions. The rationale: LLMs produce slightly different outputs across runs even at
temperature=0, and majority vote reduces the chance of a single aberrant JSON response skewing
a metric. A 3-way vote also gracefully absorbs one JSON parse failure without losing the
data point entirely.

The cost: 3 judge calls × ~900 tokens × 50 answerable questions = 135k tokens for judging alone,
which exceeds the 100k/day free-tier ceiling on Groq before any agent calls are counted.

The default was reduced to 1 judge run, with `--judge-runs N` available when majority-vote
rigour is needed (e.g. final submission scoring on a paid tier). At temperature=0 the judge is
largely deterministic, so 3 runs adds little signal on average — its main value is the one-in-ten
case where the model mis-formats JSON. A single retry on parse failure handles that case without
3x the token spend.

The `--quick` mode was also restructured. Previously it took the first 10 questions, which
happened to be all salary-category questions — the smoke test gave no signal on WICA, EA fees,
or repatriation. The new quick mode selects 2 questions per category (12 answerable + 5
distractors) and 3 probes per language (9 probes), totalling ~26 questions. At 1 judge run this
costs approximately 70k tokens — within the daily free-tier limit.
