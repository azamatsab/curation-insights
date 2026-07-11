"""The two indexes: a SQLite structured store (for aggregation/ranking) and a ChromaDB
vector store (for semantic retrieval). Kept behind a thin module so ingest and query
share one definition of the schema and collections."""
import sqlite3
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


def most_discussed(ticker, category=None, top_n=6):
    con = db()
    q = """SELECT theme, COUNT(*) n, COUNT(DISTINCT sender) voices
           FROM insights WHERE ticker = ?"""
    args = [ticker]
    if category:
        q += " AND category = ?"
        args.append(category)
    q += " GROUP BY theme ORDER BY n DESC, voices DESC LIMIT ?"
    args.append(top_n)
    rows = [dict(r) for r in con.execute(q, args)]
    con.close()
    return rows


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


def semantic_search(name, query, k=8, ticker=None):
    """Retrieve top matches. If ticker is given, over-fetch and post-filter on the
    comma-joined `tickers` metadata (avoids Chroma's scalar-only where clause)."""
    col = collection(name)
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(max(k * 5, k), n))
    out = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        if ticker and ticker not in (meta.get("tickers", "") or "").split(","):
            continue
        out.append({"text": doc, **meta})
        if len(out) >= k:
            break
    return out
