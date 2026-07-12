# Investor Insight Explorer

A conversational research assistant over a private community of investors. You ask a
question in plain English and get back **synthesised insights** drawn from the community's
own chat, and — for Tesla — cross-checked against the company's quarterly filings.

> Flagship query: **"What are the most discussed risks investors mentioned about Tesla?"**

---

## The one idea the design turns on

There are **two fundamentally different query shapes**, and a naive RAG only serves one:

| Shape | Example | What it needs |
|---|---|---|
| **Open synthesis** | "What do investors think about Tesla's margins?" | semantic retrieval + LLM synthesis |
| **Aggregation / ranking** | "What are the *most discussed* risks?" | **count & rank across the whole community** |

A top-k vector search **cannot** answer "most discussed" — it returns the chunks most
similar to the query text, not a ranking over every message. So the pipeline runs an
**offline enrichment pass**: every chat message is turned into structured *insight records*
(`ticker, theme, category, sentiment, claim`) stored in SQLite, where "most discussed risk"
is a plain `GROUP BY theme … COUNT … ORDER BY`. The vector store then supplies the wording
and the citations. This **two-index** split is the core of the design.

Why not do the ranking in the vector DB? ChromaDB can *filter* by metadata but it is not an
aggregation engine — `COUNT`, `GROUP BY`, `COUNT(DISTINCT author)` over the full corpus is an
OLAP job, which is exactly what the relational store is for.

---

## Architecture

```
 OFFLINE (ingest.py)                              ONLINE (query.py / app.py)
 ───────────────────                              ──────────────────────────
 interview_synthetic_chat.json                    user question
        │                                                │
   per-message LLM extraction  ── cached ──►      router (ticker + intent)
        │  {ticker,theme,category,                       │
        │   sentiment,claim}                    ┌────────┴─────────┐
        ├──► SQLite  (structured index)         ▼ aggregate        ▼ synthesis
        └──► Chroma  (vector index, chat)   SQLite GROUP BY    Chroma semantic
                                            (rank themes)      search (chat + filings)
 tesla/*.pdf                                        └────────┬─────────┘
   text → overlapping chunks ──► Chroma (filings)            ▼
        │                                          LLM synthesis + citations
   per-chunk LLM risk extraction ── cached ──►               ▼
        {theme, claim} per quarter ──► SQLite     answer  ·  latency  ·  cost
        (company-disclosed risk themes)
```

- **Structured index — SQLite.** One row per insight → aggregation / ranking.
  Ranking is **recency-weighted**: each insight decays by a half-life
  (`RECENCY_HALF_LIFE_DAYS`), so "most discussed" means "most discussed *lately*" and a
  stale burst can't dominate. `n` (raw count) and `voices` (distinct investors) are shown
  alongside the weighted `score`. Reference "now" is the newest timestamp in the data.
- **Vector index — ChromaDB.** Chat messages + Tesla filing chunks → semantic evidence.
  Embeddings are the **local** default (all-MiniLM via onnxruntime) — no API key, no rate
  limits, so retrieval never depends on a paid API being up. A deliberate reliability choice.
  Retrieval is **hybrid**: dense (Chroma) + lexical (BM25) fused with Reciprocal Rank
  Fusion. BM25 nails exact cashtags / tickers / numbers that dense embeddings blur — which
  matters here because the chat is full of `$TSLA`-style symbols. Toggle with `HYBRID=0`.
- **LLM (Anthropic).** Used in exactly two places: offline insight extraction and answer
  synthesis. Default model **Haiku 4.5** (cheapest capable); override to Opus via env.

For Tesla, the flagship query becomes a strong pairing: the community's **most-discussed
risks** next to the risks **Tesla itself reports** in its quarterly updates. The filing
side is pre-structured too — a second enrichment pass turns each filing chunk into
`(theme, claim)` rows per quarter (416 disclosures across 12 quarters), using the same
theme vocabulary as the chat extraction. That makes "community themes vs company-disclosed
themes" an **exact join, not a semantic guess**, and the answer can say things like
*"production is Tesla's most-disclosed risk (121 disclosures, every quarter Q1'22–Q4'24)
yet barely appears in community discussion."*

---

## Run it

```bash
pip install -r requirements.txt          # anthropic, chromadb, pymupdf, streamlit
export ANTHROPIC_API_KEY=sk-ant-...       # or cp .env.example .env and fill it in
python ingest.py                          # builds SQLite + Chroma (idempotent, cached)
python ingest.py --filing-risks           # structured risk extraction from filings
#                 --sample 10             #   (validate the prompt on 10 chunks first)
streamlit run app.py                      # chat UI
# or headless:
python query.py "What are the most discussed risks investors mentioned about Tesla?"
```

`ingest.py` is **idempotent** — extraction is cached by message hash in
`store/extract_cache.jsonl`, so a re-run costs nothing and only newly added messages hit
the API. Ingesting the 1000 messages from scratch takes ~3 minutes and ~$0.43.

---

## How it maps to the product's primary metrics

| Metric | Where it shows up |
|---|---|
| **# tickers retrievable** | sidebar "Tickers retrievable" + full list (from the structured index) |
| **# themes discoverable** | sidebar "Themes" panel per ticker = the aggregation output |
| **Response time** | latency shown under every answer + sidebar average |
| **# questions sent / retention** | every answered query is logged to `store/usage_log.jsonl` (timestamp, session id, route, latency, cost); sidebar shows questions sent + distinct sessions |
| **Answer satisfaction** | 👍/👎 under every answer (`st.feedback`) → same log; if a user changes their mind, the last verdict per (session, question) wins; sidebar shows the tally |

---

## Cost control

- **Haiku 4.5** for the high-volume extraction (~$0.43 for all 1000 messages; ~$0.27 for
  the 355 filing chunks).
- **Batched** extraction (20 messages / 8 chunks per call) amortizes the prompt.
- **Cache-on-write** so re-ingestion never re-spends.
- **Hard budget guard** (`BUDGET_USD`, default $2.50): extraction aborts before overrun.
- Per-call token usage + $ logged to `store/cost_log.jsonl`; the UI shows per-answer cost.

---

## Beyond the core

Originally cut to fit the time box, **now built**:

- **Structured extraction from filings** — a filing-side risk table (`filing_risks`: theme,
  claim, quarter) built with the same cached, budget-guarded pass as the chat extraction.
  Makes the community-vs-company comparison exact rather than semantic, and enables
  quarter-over-quarter views of what Tesla itself keeps flagging.
- **Stronger faithfulness judge** — the eval judge is now a stronger model
  (`JUDGE_MODEL`, default Sonnet) with a majority vote (`JUDGE_VOTES=3`). The cheap Haiku
  judge false-positived on real citations; the stronger judge instead catches *real*
  paraphrase/attribution slips (see Evaluation below).
- **Usage-metric logging** — every answered query appends to `store/usage_log.jsonl`
  (session, route, latency, cost), surfaced in the sidebar — plus a direct satisfaction
  signal: 👍/👎 on each answer, logged to the same file.

**Retrieval + reliability features** that pay off on this data:

- **Recency-weighted ranking** — half-life decay on the aggregation (`RECENCY_HALF_LIFE_DAYS`).
- **Hybrid retrieval** — dense + BM25 via RRF, so exact cashtags/tickers aren't lost to embeddings.
- **Hybrid router** — keyword fast-path + a cheap LLM classifier on ambiguous queries (fixes vague
  "what about X?" going thin); toggle with `LLM_ROUTER=0`.
- **Cross-ticker aggregation** — aggregate queries *without* a ticker ("where's the biggest
  opportunity?") rank tickers across the whole community (`GROUP BY ticker`, recency-weighted,
  distinct voices) instead of falling back to blind top-k retrieval.
- **Corrective retrieval (CRAG-light)** — when a ticker's evidence comes back thin, a second
  ticker-anchored retrieval pass, at no extra LLM cost.
- **Streaming synthesis** — answers stream token-by-token in the UI; same cost, far lower
  *perceived* latency (time-to-first-token ~1s vs ~6s for the full answer).

### Evaluation — `python eval.py`

Answers "how do you know it works?" on a small gold set, reusing the built indexes
(~$0.30/run, most of it the Sonnet judge; set `JUDGE_MODEL=claude-haiku-4-5 JUDGE_VOTES=1`
for a ~$0.05 quick pass):

| Metric | What it measures | Latest |
|---|---|---|
| **Routing accuracy** | did the router pick the right ticker + intent? (deterministic) | 6/6 |
| **Citation grounding** | is every `[chat:X]` / `[filing:Y]` the answer cites actually in the retrieved evidence? (deterministic — the reliable faithfulness signal) | 6/6 |
| **LLM-judge faithfulness** | does a Sonnet judge (majority of 3) find materially unsupported claims? (advisory) | 2–3/6 across runs |

The judge story is itself a finding, in two acts. A cheap single-shot Haiku judge was
noise: it false-positived on real citations and swung 50%→17%→33% across minor prompt
edits. The upgraded judge (stronger model, majority vote, fabrication-only criteria) is
*consistent* — and what it now flags are genuine synthesis slips (e.g. a quote attributed
to the wrong sender, or "struggling to see the **bear** case" summarized as bearish
sentiment). So the harness anchors on the deterministic grounding check for citations and
uses the judge as a real, actionable signal on paraphrase quality — the remaining gap is
honest headroom, not judge noise. (The judge number varies 2–3/6 between runs because the
*synthesis* is stochastic — each run regenerates the answers; the deterministic anchors
are stable at 6/6.)

---

## Layout

```
config.py    models, paths, prices, budget guard, company-name map
llm.py       Anthropic client: batched extraction (chat + filing risks), synthesis, cost log
store.py     SQLite (aggregation: insights + filing_risks) + Chroma (vector) — the two indexes
ingest.py    offline write path: chat + tesla -> the two stores; --filing-risks pass
query.py     online read path: route -> aggregate/retrieve -> synthesize; usage logging
app.py       Streamlit chat UI + coverage/usage sidebar
data/        interview_synthetic_chat.json, tesla/*.pdf
store/        built indexes + caches + cost/usage logs (gitignored)
```
