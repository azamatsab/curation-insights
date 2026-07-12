"""Conversational UI. Run:  streamlit run app.py

Sidebar surfaces the two product metrics directly: tickers retrievable and themes
discoverable. Each answer shows its latency and $ cost.
"""
import uuid

import streamlit as st

import store
import query

st.set_page_config(page_title="Curation — Investor Insights", page_icon="🔎", layout="wide")

st.title("🔎 Investor Insight Explorer")
st.caption("Ask about any stock or token the community discusses. Tesla answers are "
           "augmented with the company's own quarterly filings.")

# ------------------------------------------------------------------ sidebar (coverage)
cov = store.ticker_coverage()
usage = query.usage_stats()
with st.sidebar:
    st.header("Coverage")
    st.metric("Tickers retrievable", len(cov))
    st.metric("Insights indexed", sum(c["n"] for c in cov))
    st.divider()
    st.header("Usage")
    u1, u2 = st.columns(2)
    u1.metric("Questions sent", usage["questions"])
    u2.metric("Sessions", usage["sessions"])
    st.caption(f"avg response time: {usage['avg_latency_s']}s")
    if usage["fb_up"] + usage["fb_down"]:
        st.caption(f"answer feedback: 👍 {usage['fb_up']} · 👎 {usage['fb_down']}")
    st.divider()
    st.subheader("Tickers")
    labels = [f'{c["ticker"]}  ·  {c["n"]}' for c in cov]
    pick = st.radio("Pick a ticker to see its themes", labels, index=0) if cov else None
    if pick:
        tk = pick.split("  ·")[0].strip()
        st.subheader(f"Themes discoverable · {tk}")
        for th in store.themes_for(tk)[:12]:
            st.write(f"**{th['theme']}** — {th['n']}  "
                     f"(🔻{th['risks']} risk / 🟢{th['opps']} opp)")

# ---------------------------------------------------------------------- chat state
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

st.markdown("**Try:** *What are the most discussed risks investors mentioned about Tesla?*")


def _log_feedback(i):
    """on_change callback for the thumbs widget under history turn `i`."""
    val = st.session_state.get(f"fb_{i}")
    if val is not None:
        turn = st.session_state.history[i]
        query.log_feedback(st.session_state.session_id, turn.get("q"),
                           "up" if val == 1 else "down")


for i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            st.feedback("thumbs", key=f"fb_{i}", on_change=_log_feedback, args=(i,))

if q := st.chat_input("Ask about a stock or token…"):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching community + filings…"):
            gen, res = query.answer_stream(q, session=st.session_state.session_id)
        st.write_stream(gen)                    # stream synthesis token-by-token
        r = res["route"]
        st.caption(f"⏱ {res['latency_s']}s · 💸 ${res['cost_usd']} · "
                   f"route: {r['intent']}" + (f" · {r['ticker']}" if r["ticker"] else "")
                   + (f" · {r['category']}" if r["category"] else "")
                   + (" · 🤖 llm-router" if r.get("llm_router") else ""))
        if res["ranked"]:
            with st.expander("Ranked themes (from the structured index)"):
                for x in res["ranked"]:
                    st.write(f"**{x['theme']}** — {x['n']} mentions · {x['voices']} investors")
        if res.get("ranked_tickers"):
            with st.expander("Ranked tickers (from the structured index)"):
                for x in res["ranked_tickers"]:
                    label = f"{x['ticker']} ({x['name']})" if x.get("name") else x["ticker"]
                    st.write(f"**{label}** — {x['n']} mentions · {x['voices']} investors")
        if res.get("company_risks"):
            with st.expander("What Tesla itself discloses (structured from filings)"):
                for x in res["company_risks"]:
                    st.write(f"**{x['theme']}** — {x['n']} disclosures · {', '.join(x['quarters'])}")
        srcs = res["chat_sources"] + res["filing_sources"]
        if srcs:
            with st.expander(f"Sources ({len(srcs)})"):
                for s in res["chat_sources"]:
                    st.write(f"💬 **{s.get('sender','?')}** — {s['text']}")
                for s in res["filing_sources"]:
                    st.write(f"📄 **{s.get('source','filing')}** — {s['text'][:300]}…")
        # same key this turn will get in the history loop on the next rerun
        fb_i = len(st.session_state.history)
        st.feedback("thumbs", key=f"fb_{fb_i}", on_change=_log_feedback, args=(fb_i,))
    st.session_state.history.append(
        {"role": "assistant", "content": res["answer"], "q": q})
