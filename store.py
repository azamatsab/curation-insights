"""The two indexes: a SQLite structured store (for aggregation/ranking) and a ChromaDB
vector store (for semantic retrieval). Kept behind a thin module so ingest and query
share one definition of the schema and collections.

Two retrieval-quality features live here:
  * recency-weighted aggregation  -> most_discussed(...) decays old insights
  * hybrid retrieval (dense+BM25) -> semantic_search(...) fuses vector and lexical hits
"""
import math
import re
import sqlite3
from datetime import datetime

import chromadb

import config

# ------------------------------------------------------------------ SQLite (OLAP)
# A vector DB can filter by metadata but is not an aggregation engine. "Most discussed
# risk" = GROUP BY + COUNT + COUNT(DISTINCT author) over the whole corpus, which is what
# this relational store is for.


def db():
    con = sqlite3.connect(config.SQLITE_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS messages (
        hash TEXT PRIMARY KEY,
        sender TEXT, ts TEXT, text TEXT
    );
    CREATE TABLE IF NOT EXISTS insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_hash TEXT REFERENCES messages(hash),
        ticker TEXT, asset_name TEXT, asset_type TEXT,
        theme TEXT, category TEXT, sentiment TEXT, claim TEXT,
        sender TEXT, ts TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ticker ON insights(ticker);
    CREATE INDEX IF NOT EXISTS idx_ticker_cat ON insights(ticker, category);
    """)
    con.commit()
    con.close()


# reference "now" for recency = the newest insight timestamp (deterministic, cached)
_REF_NOW = None


def _reference_now():
    global _REF_NOW
    if _REF_NOW is None:
        con = db()
        row = con.execute("SELECT MAX(ts) FROM insights").fetchone()
        con.close()
        _REF_NOW = datetime.fromisoformat(row[0]) if row and row[0] else datetime(2024, 11, 19)
    return _REF_NOW


def _recency_weight(ts):
    """Half-life decay: an insight `h` days old counts as 0.5**(h/half_life)."""
    hl = config.RECENCY_HALF_LIFE_DAYS
    if hl <= 0:
        return 1.0
    age_days = (_reference_now() - datetime.fromisoformat(ts)).total_seconds() / 86400
    return 0.5 ** (max(age_days, 0) / hl)


def most_discussed(ticker, category=None, top_n=6):
    """Rank themes for a ticker. `n` is the raw mention count; `score` is the
    recency-weighted count (what we sort by when decay is on). `voices` = distinct
    investors, so one loud poster can't dominate."""
    con = db()
    q = "SELECT theme, sender, ts FROM insights WHERE ticker = ?"
    args = [ticker]
    if category:
        q += " AND category = ?"
        args.append(category)
    rows = con.execute(q, args).fetchall()
    con.close()

    agg = {}
    for r in rows:
        a = agg.setdefault(r["theme"], {"theme": r["theme"], "n": 0, "score": 0.0, "voices": set()})
        a["n"] += 1
        a["score"] += _recency_weight(r["ts"])
        a["voices"].add(r["sender"])
    out = [{"theme": a["theme"], "n": a["n"], "score": round(a["score"], 2),
            "voices": len(a["voices"])} for a in agg.values()]
    key = "score" if config.RECENCY_HALF_LIFE_DAYS > 0 else "n"
    out.sort(key=lambda x: (x[key], x["voices"]), reverse=True)
    return out[:top_n]


def ticker_coverage():
    con = db()
    rows = [dict(r) for r in con.execute(
        """SELECT ticker, asset_type, COUNT(*) n, COUNT(DISTINCT sender) voices
           FROM insights GROUP BY ticker ORDER BY n DESC""")]
    con.close()
    return rows


def themes_for(ticker):
    con = db()
    rows = [dict(r) for r in con.execute(
        """SELECT theme,
                  SUM(category='risk') risks,
                  SUM(category='opportunity') opps,
                  COUNT(*) n
           FROM insights WHERE ticker = ? GROUP BY theme ORDER BY n DESC""", (ticker,))]
    con.close()
    return rows


def known_tickers():
    con = db()
    rows = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM insights")]
    con.close()
    return rows


# ----------------------------------------------------------------- Chroma (vector)
def _chroma():
    return chromadb.PersistentClient(path=config.CHROMA_PATH)


def collection(name):
    # Default embedding function = local all-MiniLM (onnxruntime). No API key needed.
    return _chroma().get_or_create_collection(name)


# ---- BM25 lexical index, built lazily from the same docs Chroma already stores ----
_BM25 = {}  # name -> {"bm25", "ids", "lut"}


def _tok(text):
    # keep $tickers and numbers so lexical match nails cashtags dense vectors blur
    return re.findall(r"[a-z0-9$]+", text.lower())


def _bm25_index(name):
    if name not in _BM25:
        from rank_bm25 import BM25Okapi
        got = collection(name).get(include=["documents", "metadatas"])
        ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
        _BM25[name] = {
            "bm25": BM25Okapi([_tok(d) for d in docs]) if docs else None,
            "ids": ids,
            "lut": {i: (d, m) for i, d, m in zip(ids, docs, metas)},
        }
    return _BM25[name]


def _passes_ticker(meta, ticker):
    return (not ticker) or ticker in (meta.get("tickers", "") or "").split(",")


def semantic_search(name, query, k=8, ticker=None):
    """Hybrid retrieval: dense (Chroma) + lexical (BM25), fused with Reciprocal Rank
    Fusion. Falls back to dense-only if HYBRID is off. Ticker filtering is a post-filter
    on the comma-joined `tickers` metadata (Chroma's where clause is scalar-only)."""
    col = collection(name)
    if col.count() == 0:
        return []

    # dense ranks
    dres = col.query(query_texts=[query], n_results=min(config.DENSE_POOL, col.count()))
    dense_ids = dres["ids"][0]

    if not config.HYBRID:
        lut = {i: (d, m) for i, d, m in zip(dres["ids"][0], dres["documents"][0], dres["metadatas"][0])}
        fused = dense_ids
    else:
        idx = _bm25_index(name)
        lut = idx["lut"]
        # sparse ranks
        sparse_ids = []
        if idx["bm25"] is not None:
            scores = idx["bm25"].get_scores(_tok(query))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:config.DENSE_POOL]
            sparse_ids = [idx["ids"][i] for i in order]
        # reciprocal rank fusion
        rrf = {}
        for rank, i in enumerate(dense_ids):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        for rank, i in enumerate(sparse_ids):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        fused = sorted(rrf, key=rrf.get, reverse=True)

    out = []
    for i in fused:
        if i not in lut:
            continue
        doc, meta = lut[i]
        if not _passes_ticker(meta, ticker):
            continue
        out.append({"text": doc, **meta})
        if len(out) >= k:
            break
    return out
