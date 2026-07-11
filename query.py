"""Online query path (read path).

    question -> router (ticker + intent) -> { structured aggregation | vector retrieval }
             -> LLM synthesis with citations -> answer

The router is a plain keyword function, not an LLM call — one fewer round-trip on the
latency budget. Swapping in an LLM classifier for ambiguous queries is a drop-in upgrade.
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
    # explicit uppercase symbol in the query
    for tok in re.findall(r"\b[A-Z]{2,5}\b", q):
        if tok in known:
            return tok
    # company name -> ticker
    for tk, name in config.COMPANY_NAMES.items():
        if tk in known and any(w in ql for w in name.split()):
            return tk
    return None


def route(q, known):
    ql = q.lower()
    intent = "aggregate" if any(w in ql for w in AGG_TRIGGERS) else "synthesis"
    category = None
    if any(w in ql for w in RISK_WORDS):
        category = "risk"
    elif any(w in ql for w in OPP_WORDS):
        category = "opportunity"
    return {"ticker": detect_ticker(q, known), "intent": intent, "category": category}


def answer(q):
    t0 = time.time()
    known = set(store.known_tickers())
    r = route(q, known)
    ticker, want_filings = r["ticker"], r["ticker"] in config.FILING_TICKERS

    ranked = None
    if r["intent"] == "aggregate" and ticker:
        ranked = store.most_discussed(ticker, category=r["category"], top_n=6)

    # targeted retrieval: seed the query with the ranked themes so the vector search
    # fetches evidence FOR those themes, not a blind top-k.
    seed = q + (" " + " ".join(x["theme"] for x in ranked) if ranked else "")
    chat_ev = store.semantic_search("chat", seed, k=8, ticker=ticker)
    filing_ev = store.semantic_search("filings", seed, k=5) if want_filings else []

    text, cost = llm.synthesize(q, ranked=ranked, chat_evidence=chat_ev,
                                filing_evidence=filing_ev)
    return {
        "answer": text,
        "route": r,
        "ranked": ranked or [],
        "chat_sources": chat_ev,
        "filing_sources": filing_ev,
        "latency_s": round(time.time() - t0, 2),
        "cost_usd": round(cost, 5),
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the most discussed risks investors mentioned about Tesla?"
    res = answer(q)
    print(f"\nQ: {q}\n")
    print(res["answer"])
    print(f"\n[route={res['route']}  latency={res['latency_s']}s  cost=${res['cost_usd']}]")
    if res["ranked"]:
        print("ranked themes:", [(x["theme"], x["n"], x["voices"]) for x in res["ranked"]])
