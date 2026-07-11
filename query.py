"""Online query path (read path).

    question -> router (ticker + intent) -> { structured aggregation | vector retrieval }
             -> corrective retry if weak -> LLM synthesis with citations -> answer

Router is keyword-first and escalates to a cheap LLM classifier only when it's unsure
(no ticker, or a vague "what about X?"). Retrieval is hybrid (dense + BM25) and, when a
ticker's evidence comes back thin, does a light corrective second pass seeded with that
ticker's top themes — a no-extra-LLM-cost version of corrective RAG.
"""
import re
import time

import config
import store
import llm

AGG_TRIGGERS = ("most", "top", "biggest", "which", "ranked", "common",
                "how many", "list", "discussed", "popular")
RISK_WORDS = ("risk", "bear", "concern", "downside", "worry", "threat")
OPP_WORDS = ("opportunity", "opportunit", "bull", "upside", "catalyst", "buy")


def detect_ticker(q, known):
    ql = q.lower()
    for tok in re.findall(r"\b[A-Z]{2,5}\b", q):        # explicit symbol
        if tok in known:
            return tok
    for tk, name in config.COMPANY_NAMES.items():        # company name -> ticker
        if tk in known and any(w in ql for w in name.split()):
            return tk
    return None


def route(q, known):
    ql = q.lower()
    intent = "aggregate" if any(w in ql for w in AGG_TRIGGERS) else "synthesis"
    category = "risk" if any(w in ql for w in RISK_WORDS) else \
               "opportunity" if any(w in ql for w in OPP_WORDS) else None
    ticker = detect_ticker(q, known)
    used_llm = False

    # keyword router is low-confidence when it can't find a ticker, or the query is a
    # vague ticker mention with no explicit intent/category (e.g. "what about Tesla?").
    ambiguous = ticker is None or (intent == "synthesis" and category is None
                                   and len(re.findall(r"\w+", q)) <= 5)
    if config.LLM_ROUTER and ambiguous:
        r = llm.route_llm(q, sorted(known))
        ticker = r["ticker"] or ticker
        intent = r["intent"]
        category = r["category"] or category
        used_llm = True

    return {"ticker": ticker, "intent": intent, "category": category, "llm_router": used_llm}


def _prepare(q):
    """Shared route + retrieve step for both blocking and streaming answers."""
    t0 = time.time()
    known = set(store.known_tickers())
    r = route(q, known)
    ticker, want_filings = r["ticker"], r["ticker"] in config.FILING_TICKERS

    ranked = store.most_discussed(ticker, category=r["category"], top_n=6) \
        if r["intent"] == "aggregate" and ticker else None

    # targeted retrieval: seed the query with themes so the vector search fetches evidence
    # FOR those themes, not a blind top-k.
    if ranked:
        seed = q + " " + " ".join(x["theme"] for x in ranked)
    elif ticker:  # synthesis with a ticker -> enrich with that ticker's top themes
        seed = q + " " + " ".join(t["theme"] for t in store.themes_for(ticker)[:3])
    else:
        seed = q
    chat_ev = store.semantic_search("chat", seed, k=8, ticker=ticker)

    # corrective retry (CRAG-light, no extra LLM): if a ticker's evidence is thin, retry
    # with a broader ticker-anchored query.
    if ticker and len(chat_ev) < 4:
        chat_ev = store.semantic_search("chat", f"{ticker} {q}", k=8, ticker=ticker)

    filing_ev = store.semantic_search("filings", seed, k=5) if want_filings else []
    return {"route": r, "ticker": ticker, "ranked": ranked or [],
            "chat_sources": chat_ev, "filing_sources": filing_ev, "_t0": t0}


def answer(q):
    """Blocking answer — CLI + eval harness."""
    m = _prepare(q)
    text, cost = llm.synthesize(q, ranked=m["ranked"] or None,
                                chat_evidence=m["chat_sources"], filing_evidence=m["filing_sources"])
    return {"answer": text, "route": m["route"], "ranked": m["ranked"],
            "chat_sources": m["chat_sources"], "filing_sources": m["filing_sources"],
            "latency_s": round(time.time() - m["_t0"], 2), "cost_usd": round(cost, 5)}


def answer_stream(q):
    """Streaming answer for the UI. Returns (generator, meta). `meta` fills in answer,
    latency and cost once the generator is fully consumed."""
    m = _prepare(q)
    meta = {"route": m["route"], "ranked": m["ranked"],
            "chat_sources": m["chat_sources"], "filing_sources": m["filing_sources"]}

    def gen():
        chunks = []
        for delta in llm.synthesize_stream(q, ranked=m["ranked"] or None,
                                           chat_evidence=m["chat_sources"],
                                           filing_evidence=m["filing_sources"]):
            chunks.append(delta)
            yield delta
        meta["answer"] = "".join(chunks)
        meta["cost_usd"] = round(llm._LAST_SYNTH_COST, 5)
        meta["latency_s"] = round(time.time() - m["_t0"], 2)

    return gen, meta


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the most discussed risks investors mentioned about Tesla?"
    res = answer(q)
    print(f"\nQ: {q}\n")
    print(res["answer"])
    print(f"\n[route={res['route']}  latency={res['latency_s']}s  cost=${res['cost_usd']}]")
    if res["ranked"]:
        print("ranked themes:", [(x["theme"], x["n"], x["voices"]) for x in res["ranked"]])
