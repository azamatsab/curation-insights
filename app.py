"""Conversational UI. Run:  streamlit run app.py

Sidebar surfaces the two product metrics directly: tickers retrievable and themes
discoverable. Each answer shows its latency and $ cost.
"""
import streamlit as st

import store
import query

st.set_page_config(page_title="Curation — Investor Insights", page_icon="🔎", layout="wide")

st.title("🔎 Investor Insight Explorer")
st.caption("Ask about any stock or token the community discusses. Tesla answers are "
           "augmented with the company's own quarterly filings.")

# ------------------------------------------------------------------ sidebar (coverage)
cov = store.ticker_coverage()
with st.sidebar:
    st.header("Coverage")
    st.metric("Tickers retrievable", len(cov))
    st.metric("Insights indexed", sum(c["n"] for c in cov))
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

st.markdown("**Try:** *What are the most discussed risks investors mentioned about Tesla?*")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if q := st.chat_input("Ask about a stock or token…"):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching community + filings…"):
            gen, res = query.answer_stream(q)   # route + retrieve
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
        srcs = res["chat_sources"] + res["filing_sources"]
        if srcs:
            with st.expander(f"Sources ({len(srcs)})"):
                for s in res["chat_sources"]:
                    st.write(f"💬 **{s.get('sender','?')}** — {s['text']}")
                for s in res["filing_sources"]:
                    st.write(f"📄 **{s.get('source','filing')}** — {s['text'][:300]}…")
    st.session_state.history.append(
        {"role": "assistant", "content": res["answer"]})
