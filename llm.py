"""Anthropic access layer: batched insight extraction, query routing, answer synthesis
(blocking + streaming), and cost tracking.

Design note: the LLM is used for offline insight extraction, an optional query-router
fallback, and answer synthesis. Embeddings are local (ChromaDB default), so retrieval and
ingestion never depend on this API being up, which keeps the demo reliable under load.
"""
import json
import time
import anthropic

import config

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

_LAST_SYNTH_COST = 0.0  # exposed so the UI can show per-answer cost after a streamed call


# ---------------------------------------------------------------- cost tracking
def _log_cost(model, usage, tag):
    pin, pout = config.PRICES.get(model, (0.0, 0.0))
    cost = (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000
    with open(config.COST_LOG, "a") as f:
        f.write(json.dumps({
            "tag": tag, "model": model,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cost_usd": round(cost, 6),
        }) + "\n")
    return cost


def total_cost():
    if not config.COST_LOG.exists():
        return 0.0
    return round(sum(json.loads(l)["cost_usd"] for l in open(config.COST_LOG) if l.strip()), 4)


# ------------------------------------------------------------- insight extraction
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "insights": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                                "asset_name": {"type": "string"},
                                "asset_type": {"type": "string",
                                               "enum": ["stock", "crypto", "etf", "macro", "other"]},
                                "theme": {"type": "string"},
                                "category": {"type": "string",
                                             "enum": ["risk", "opportunity", "opinion", "data_point", "question"]},
                                "sentiment": {"type": "string",
                                              "enum": ["bullish", "bearish", "neutral"]},
                                "claim": {"type": "string"},
                            },
                            "required": ["ticker", "asset_name", "asset_type", "theme",
                                         "category", "sentiment", "claim"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["index", "insights"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """You analyze messages from a private chat of sophisticated investors.
For each message, extract structured insights about the specific assets discussed.

Rules:
- One object per (asset) mentioned with a substantive view. A message can yield 0, 1, or many.
- If a message is pure chatter, a greeting, or mentions no specific asset, return an empty list.
- ticker: uppercase symbol (e.g. TSLA, NVDA, BTC, SOL). Infer it from context if only a name is used.
- asset_name: the human name (e.g. "Tesla", "Bitcoin").
- theme: a SHORT normalized lowercase topic, reused across messages so it aggregates well
  (e.g. "demand", "margins", "valuation", "regulation", "competition", "fsd/autonomy",
   "leadership", "supply", "macro", "liquidity", "technicals", "adoption").
- category: risk = a downside concern; opportunity = an upside driver / bull case;
  opinion = a general stance; data_point = a fact/number; question = an open question.
- sentiment: the author's stance on the asset (bullish / bearish / neutral). Distinct from
  category — someone can raise a risk while staying bullish.
- claim: one concise sentence capturing the insight, in your words.

Return results keyed by the given message index."""


def extract_batch(items):
    """items: list of (index, text). Returns {index: [insight_dict, ...]}."""
    numbered = "\n".join(f"[{i}] {t}" for i, t in items)
    resp = _retry(lambda: _client.messages.create(
        model=config.EXTRACT_MODEL,
        max_tokens=4096,
        system=EXTRACT_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": EXTRACT_SCHEMA}},
        messages=[{"role": "user", "content": f"Messages:\n{numbered}"}],
    ))
    _log_cost(config.EXTRACT_MODEL, resp.usage, "extract")
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return {r["index"]: r["insights"] for r in data["results"]}


# ------------------------------------------------------------------ query router
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["aggregate", "synthesis"]},
        "ticker": {"type": "string"},                       # "" if none/unclear
        "category": {"type": "string", "enum": ["risk", "opportunity", "none"]},
    },
    "required": ["intent", "ticker", "category"],
    "additionalProperties": False,
}


def route_llm(question, known_tickers):
    """Fallback classifier used only when the keyword router is low-confidence."""
    system = (
        "Classify an investor's question for a retrieval system. Return JSON.\n"
        "- intent: 'aggregate' if it asks for a ranking/overview/'most'/'main'/'what about X' "
        "about an asset; 'synthesis' for a specific open question.\n"
        "- ticker: the uppercase symbol the question is about, else ''. "
        f"Known tickers: {', '.join(known_tickers)}\n"
        "- category: 'risk', 'opportunity', or 'none'.\n"
        "A vague question like 'what about Tesla?' is an aggregate overview."
    )
    resp = _retry(lambda: _client.messages.create(
        model=config.EXTRACT_MODEL, max_tokens=200, system=system,
        output_config={"format": {"type": "json_schema", "schema": ROUTE_SCHEMA}},
        messages=[{"role": "user", "content": question}],
    ))
    _log_cost(config.EXTRACT_MODEL, resp.usage, "route")
    r = json.loads(next(b.text for b in resp.content if b.type == "text"))
    return {"intent": r["intent"],
            "ticker": (r["ticker"] or "").upper() or None,
            "category": None if r["category"] == "none" else r["category"]}


# --------------------------------------------------------------------- synthesis
SYNTH_SYSTEM = """You are an investment-research assistant for a community of investors.
Answer the user's question using ONLY the material provided below — never invent facts.

- Group the answer by theme where it helps.
- Cite the source of each point inline: [chat: <sender>] for community messages,
  [filing: <source>] for Tesla's own reports.
- When both are present, clearly separate "What investors are saying" from
  "What the company reports" so the reader sees opinion vs. primary source.
- You surface and organize insight; you do NOT give buy/sell advice. Keep it factual and neutral.
- Do NOT quantify how many investors hold a view ("several", "multiple", "many") unless a
  count is given in the ranked themes. Only cite a [chat: sender] that appears in the evidence,
  and only quote filing wording that literally appears in the excerpts — never paraphrase a
  filing into a specific claim it doesn't state.
- If the material doesn't answer the question, say so plainly."""


def _synth_prompt(question, ranked, chat_evidence, filing_evidence):
    parts = [f"QUESTION: {question}\n"]
    if ranked:
        parts.append("RANKED THEMES (counted across the whole community, recency-weighted):")
        for r in ranked:
            parts.append(f"- {r['theme']}: {r['n']} mentions from {r['voices']} distinct investors")
        parts.append("")
    if chat_evidence:
        parts.append("COMMUNITY MESSAGES (evidence):")
        for e in chat_evidence:
            parts.append(f"- [{e['sender']}] {e['text']}")
        parts.append("")
    if filing_evidence:
        parts.append("TESLA FILING EXCERPTS (primary source):")
        for e in filing_evidence:
            parts.append(f"- [{e['source']}] {e['text']}")
        parts.append("")
    return "\n".join(parts)


def synthesize(question, ranked=None, chat_evidence=None, filing_evidence=None):
    """Blocking synthesis — used by the CLI and the eval harness."""
    resp = _retry(lambda: _client.messages.create(
        model=config.SYNTH_MODEL, max_tokens=1200, system=SYNTH_SYSTEM,
        messages=[{"role": "user", "content": _synth_prompt(question, ranked, chat_evidence, filing_evidence)}],
    ))
    cost = _log_cost(config.SYNTH_MODEL, resp.usage, "synthesize")
    answer = "".join(b.text for b in resp.content if b.type == "text")
    return answer, cost


def synthesize_stream(question, ranked=None, chat_evidence=None, filing_evidence=None):
    """Streaming synthesis — yields text deltas as they arrive (cuts perceived latency).
    Same cost as the blocking call; the token usage is logged once the stream finishes."""
    global _LAST_SYNTH_COST
    with _client.messages.stream(
        model=config.SYNTH_MODEL, max_tokens=1200, system=SYNTH_SYSTEM,
        messages=[{"role": "user", "content": _synth_prompt(question, ranked, chat_evidence, filing_evidence)}],
    ) as stream:
        for text in stream.text_stream:
            yield text
        final = stream.get_final_message()
    _LAST_SYNTH_COST = _log_cost(config.SYNTH_MODEL, final.usage, "synthesize")


# ------------------------------------------------------------------------- utils
def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
