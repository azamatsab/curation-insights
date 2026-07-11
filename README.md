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
                                                   LLM synthesis + citations
                                                             ▼
                                              answer  ·  latency  ·  cost
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
risks** next to the risks **Tesla itself reports** in its quarterly updates.

---

## Run it

```bash
pip install -r requirements.txt          # anthropic, chromadb, pymupdf, streamlit
export ANTHROPIC_API_KEY=sk-ant-...       # or cp .env.example .env and fill it in
python ingest.py                          # builds SQLite + Chroma (idempotent, cached)
streamlit run app.py                      # chat UI
# or headless:
python query.py "What are the most discussed risks investors mentioned about Tesla?"
```

`ingest.py` is **idempotent** — extraction is cached by message hash in
`store/extract_cache.jsonl`, so re-runs cost nothing and a committed `store/` lets a
reviewer skip ingestion entirely.

---

## How it maps to the product's primary metrics

| Metric | Where it shows up |
|---|---|
| **# tickers retrievable** | sidebar "Tickers retrievable" + full list (from the structured index) |
| **# themes discoverable** | sidebar "Themes" panel per ticker = the aggregation output |
| **Response time** | latency shown under every answer |
| **# questions sent / retention** | would be wired via query logging + sessions (out of MVP scope) |

---

## Cost control

- **Haiku 4.5** for the high-volume extraction (~$0.50 for all 1000 messages).
- **Batched** extraction (20 messages/call) amortizes the prompt → ~50 calls, not 1000.
- **Cache-on-write** so re-ingestion never re-spends.
- **Hard budget guard** (`BUDGET_USD`, default $2.50): extraction aborts before overrun.
- Per-call token usage + $ logged to `store/cost_log.jsonl`; the UI shows per-answer cost.

---

## Deliberately out of MVP scope (designed, not built)

Still cut to fit the time box:

- **Structured extraction from filings** — today filings are vector-only; the community is where
  aggregation matters, so structure is spent there (you pre-structure the *predictable* queries,
  and let the vector tail generalize). A filing-side risk-factor table would make the
  community-vs-company comparison exact rather than semantic.
- **Stronger faithfulness judge** — the eval's LLM judge is noisy at Haiku tier (false-positives
  on real citations); a stronger judge + majority vote would give a trustworthy claim-support number.
- **Usage-metric logging** — query logging + sessions for the "questions sent / retention" metrics.

**Built beyond the core** (retrieval + reliability features that pay off on this data):

- **Recency-weighted ranking** — half-life decay on the aggregation (`RECENCY_HALF_LIFE_DAYS`).
- **Hybrid retrieval** — dense + BM25 via RRF, so exact cashtags/tickers aren't lost to embeddings.
- **Hybrid router** — keyword fast-path + a cheap LLM classifier on ambiguous queries (fixes vague
  "what about X?" going thin); toggle with `LLM_ROUTER=0`.
- **Corrective retrieval (CRAG-light)** — when a ticker's evidence comes back thin, a second
  ticker-anchored retrieval pass, at no extra LLM cost.
- **Streaming synthesis** — answers stream token-by-token in the UI; same cost, far lower
  *perceived* latency (time-to-first-token ~1s vs ~6s for the full answer).

### Evaluation — `python eval.py`

Answers "how do you know it works?" on a small gold set, reusing the built indexes (~$0.04):

| Metric | What it measures | Latest |
|---|---|---|
| **Routing accuracy** | did the router pick the right ticker + intent? (deterministic) | 6/6 |
| **Citation grounding** | is every `[chat:X]` / `[filing:Y]` the answer cites actually in the retrieved evidence? (deterministic — the reliable faithfulness signal) | 6/6 |
| **LLM-judge faithfulness** | does an LLM judge think every claim is supported? (advisory — noisy at Haiku tier) | 2/6 |

The gap between the two faithfulness columns is itself a finding: a cheap LLM judge
false-positives on real citations, so the harness anchors on the deterministic check and
treats the judge as advisory.

---

## Layout

```
config.py    models, paths, prices, budget guard, company-name map
llm.py       Anthropic client: batched extraction, synthesis, cost logging
store.py     SQLite (aggregation) + Chroma (vector) — the two indexes
ingest.py    offline write path: chat + tesla -> the two stores
query.py     online read path: route -> aggregate/retrieve -> synthesize
app.py       Streamlit chat UI + coverage sidebar
data/        interview_synthetic_chat.json, tesla/*.pdf
store/        built indexes + caches + cost log (gitignored)
```
