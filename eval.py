"""Evaluation harness. Answers the interviewer's real question: "how do you know it works?"

Two axes:
  1. Routing accuracy  — did the router pick the right ticker + intent? (deterministic check)
  2. Answer faithfulness — does the synthesized answer make ONLY claims supported by the
     retrieved evidence? (LLM-as-judge — the automated version of the manual grep-check.)

Run:  python eval.py
Keeps a small gold set so it costs ~$0.05 to run. Reuses the built indexes (no re-ingest).
"""
import json
import re

import config
import query
import llm

# Small gold set. `ticker`/`intent` are the routing targets; None = don't care.
GOLD = [
    {"q": "What are the most discussed risks investors mentioned about Tesla?",
     "ticker": "TSLA", "intent": "aggregate"},
    {"q": "What do investors think about Tesla's valuation?",
     "ticker": "TSLA", "intent": "synthesis"},
    {"q": "What about Tesla?",                         # the vague one that used to go thin
     "ticker": "TSLA", "intent": "aggregate"},
    {"q": "What are the top opportunities investors see in NVDA?",
     "ticker": "NVDA", "intent": "aggregate"},
    {"q": "Which risks are investors flagging on Bitcoin?",
     "ticker": "BTC", "intent": "aggregate"},
    {"q": "What is the bull case for Solana?",
     "ticker": "SOL", "intent": "synthesis"},
]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "note": {"type": "string"},
    },
    "required": ["faithful", "unsupported_claims", "note"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You are a strict fact-checker. You are given an ANSWER and the exact
EVIDENCE it was allowed to use. Decide whether every substantive claim in the answer is
supported by the evidence. A claim that generalizes, adds numbers, or names a source not in
the evidence is NOT supported. Ignore hedging and generic framing. Return faithful=false and
list any unsupported claims."""


def judge(answer, chat_sources, filing_sources, ranked):
    evidence = []
    for r in ranked:
        evidence.append(f"[ranked theme] {r['theme']}: {r['n']} mentions, {r['voices']} investors")
    for s in chat_sources:
        evidence.append(f"[chat: {s.get('sender','?')}] {s['text']}")
    for s in filing_sources:
        evidence.append(f"[filing: {s.get('source','?')}] {s['text']}")
    prompt = f"ANSWER:\n{answer}\n\nEVIDENCE:\n" + "\n".join(evidence)
    resp = llm._client.messages.create(
        model=config.EXTRACT_MODEL, max_tokens=600, system=JUDGE_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    llm._log_cost(config.EXTRACT_MODEL, resp.usage, "judge")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def citation_grounding(answer, chat_sources, filing_sources):
    """Deterministic faithfulness proxy: every [chat: X] / [filing: Y] the answer cites
    must actually be in the retrieved evidence. Reliable (no model), so it anchors the
    metric while the LLM judge is treated as advisory."""
    chat_senders = {s.get("sender") for s in chat_sources}
    filing_srcs = {s.get("source") for s in filing_sources}
    cited_chat = {c.strip() for grp in re.findall(r"\[chat:\s*([^\]]+)\]", answer)
                  for c in grp.split(",")}
    cited_filing = {c.strip() for c in re.findall(r"\[filing:\s*([^\]]+)\]", answer)}
    bad_chat = [c for c in cited_chat if c and c not in chat_senders]
    bad_filing = [c for c in cited_filing if c and c not in filing_srcs]
    return (not bad_chat and not bad_filing), bad_chat + bad_filing


def main():
    start_cost = llm.total_cost()
    route_ok, ground_ok, faith_ok, lat = 0, 0, 0, []
    print(f"Running eval on {len(GOLD)} queries "
          f"(HYBRID={config.HYBRID}, LLM_ROUTER={config.LLM_ROUTER}, "
          f"half_life={config.RECENCY_HALF_LIFE_DAYS})\n")

    for g in GOLD:
        res = query.answer(g["q"])
        r = res["route"]
        r_ok = (g["ticker"] is None or r["ticker"] == g["ticker"]) and \
               (g["intent"] is None or r["intent"] == g["intent"])
        grounded, bad = citation_grounding(res["answer"], res["chat_sources"], res["filing_sources"])
        v = judge(res["answer"], res["chat_sources"], res["filing_sources"], res["ranked"])
        route_ok += r_ok
        ground_ok += grounded
        faith_ok += v["faithful"]
        lat.append(res["latency_s"])

        print(f"Q: {g['q']}")
        print(f"   route: got ({r['ticker']}, {r['intent']}) want ({g['ticker']}, {g['intent']})"
              f"  ->  {'OK' if r_ok else 'MISS'}"
              + ("  [llm-router]" if r.get("llm_router") else ""))
        print(f"   citations grounded: {grounded}" + (f"  ⚠ fabricated: {bad}" if not grounded else ""))
        print(f"   judge faithful (advisory): {v['faithful']}")
        print(f"   sources: {len(res['chat_sources'])} chat / {len(res['filing_sources'])} filing"
              f"  ·  {res['latency_s']}s\n")

    n = len(GOLD)
    print("=" * 60)
    print(f"Routing accuracy        : {route_ok}/{n}  ({route_ok / n:.0%})")
    print(f"Citation grounding      : {ground_ok}/{n}  ({ground_ok / n:.0%})   [deterministic — primary]")
    print(f"LLM-judge faithfulness  : {faith_ok}/{n}  ({faith_ok / n:.0%})   [advisory — noisy at Haiku tier]")
    print(f"Avg latency             : {sum(lat) / n:.1f}s")
    print(f"Eval cost               : ${llm.total_cost() - start_cost:.4f}")


if __name__ == "__main__":
    main()
