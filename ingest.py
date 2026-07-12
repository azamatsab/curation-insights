"""Offline ingestion pipeline (write path).

  chat.json   -> per-message LLM insight extraction (batched, cached) -> SQLite + Chroma
  tesla/*.pdf -> text extraction -> overlapping chunks              -> Chroma

Idempotent: extraction is cached by message hash, so re-runs are free and never
re-spend on the API. Run:  python ingest.py
"""
import hashlib
import json
import re
import sys

import config
import store
import llm


def _hash(m):
    return hashlib.md5(f"{m['sender']}|{m['datetime']}|{m['message']}".encode()).hexdigest()[:16]


# ------------------------------------------------------------------ extraction cache
def _load_cache():
    cache = {}
    if config.EXTRACT_CACHE.exists():
        for line in open(config.EXTRACT_CACHE):
            if line.strip():
                r = json.loads(line)
                cache[r["hash"]] = r["insights"]
    return cache


def _append_cache(h, insights):
    with open(config.EXTRACT_CACHE, "a") as f:
        f.write(json.dumps({"hash": h, "insights": insights}) + "\n")


# ------------------------------------------------------------------------- chat
def ingest_chat():
    msgs = json.load(open(config.CHAT_JSON))
    for m in msgs:
        m["hash"] = _hash(m)
    cache = _load_cache()

    todo = [m for m in msgs if m["hash"] not in cache]
    print(f"chat: {len(msgs)} messages, {len(todo)} need extraction "
          f"({len(msgs) - len(todo)} cached)")

    for start in range(0, len(todo), config.EXTRACT_BATCH):
        if llm.total_cost() >= config.BUDGET_USD:
            print(f"!! budget guard hit (${llm.total_cost()} >= ${config.BUDGET_USD}) — stopping extraction")
            sys.exit(1)
        batch = todo[start:start + config.EXTRACT_BATCH]
        items = [(i, batch[i]["message"]) for i in range(len(batch))]
        result = llm.extract_batch(items)
        for i, m in enumerate(batch):
            _append_cache(m["hash"], result.get(i, []))
        print(f"  extracted {start + len(batch)}/{len(todo)}  "
              f"(spend so far: ${llm.total_cost()})")

    # rebuild indexes from the (now complete) cache
    cache = _load_cache()
    store.init_db()
    con = store.db()
    con.execute("DELETE FROM insights")
    con.execute("DELETE FROM messages")
    col = store.collection("chat")
    if col.count():
        store._chroma().delete_collection("chat")
        col = store.collection("chat")

    ids, docs, metas = [], [], []
    for m in msgs:
        con.execute("INSERT OR REPLACE INTO messages VALUES (?,?,?,?)",
                    (m["hash"], m["sender"], m["datetime"], m["message"]))
        insights = cache.get(m["hash"], [])
        tickers = sorted({ins["ticker"].upper() for ins in insights if ins.get("ticker")})
        for ins in insights:
            con.execute(
                """INSERT INTO insights
                   (msg_hash,ticker,asset_name,asset_type,theme,category,sentiment,claim,sender,ts)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (m["hash"], ins["ticker"].upper(), ins.get("asset_name", ""),
                 ins.get("asset_type", ""), ins.get("theme", "").lower(),
                 ins.get("category", ""), ins.get("sentiment", ""), ins["claim"],
                 m["sender"], m["datetime"]))
        if insights:  # only embed messages that carry a real insight
            ids.append(m["hash"])
            docs.append(m["message"])
            metas.append({"sender": m["sender"], "date": m["datetime"],
                          "tickers": ",".join(tickers)})
    con.commit()
    con.close()
    for s in range(0, len(ids), 500):
        col.add(ids=ids[s:s + 500], documents=docs[s:s + 500], metadatas=metas[s:s + 500])
    print(f"chat: wrote {len(ids)} messages to vector store; insights -> SQLite")


# ------------------------------------------------------------------------ tesla
def _chunk(text, size=1400, overlap=200):
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def ingest_tesla():
    import fitz  # pymupdf
    pdfs = sorted(config.TESLA_DIR.glob("*.pdf"))
    print(f"tesla: {len(pdfs)} filings")
    col = store.collection("filings")
    if col.count():
        store._chroma().delete_collection("filings")
        col = store.collection("filings")

    ids, docs, metas = [], [], []
    for pdf in pdfs:
        m = re.search(r"Q(\d)-(\d{4})", pdf.name)
        quarter, year = (f"Q{m.group(1)}", m.group(2)) if m else ("", "")
        text = ""
        with fitz.open(pdf) as doc:
            for page in doc:
                text += page.get_text()
        chunks = _chunk(text)
        for ci, ch in enumerate(chunks):
            if len(ch) < 120:
                continue
            ids.append(f"{pdf.stem}_{ci}")
            docs.append(ch)
            metas.append({"source": f"{quarter} {year}".strip() or pdf.stem,
                          "file": pdf.name, "tickers": "TSLA"})
        print(f"  {pdf.name}: {len(chunks)} chunks")
    for s in range(0, len(ids), 500):
        col.add(ids=ids[s:s + 500], documents=docs[s:s + 500], metadatas=metas[s:s + 500])
    print(f"tesla: wrote {len(ids)} chunks to vector store")


# ------------------------------------------------- structured filing risk extraction
def _load_filing_cache():
    cache = {}
    if config.FILING_EXTRACT_CACHE.exists():
        for line in open(config.FILING_EXTRACT_CACHE):
            if line.strip():
                r = json.loads(line)
                cache[r["chunk_id"]] = r["risks"]
    return cache


def extract_filing_risks(sample=None):
    """Second enrichment pass: turn Tesla's filing chunks (already in Chroma) into
    structured (theme, claim) rows in SQLite, so community-vs-company theme comparison
    is exact. Cached per chunk id; `sample=N` processes only the first N uncached chunks
    (validate the prompt cheaply before the full run)."""
    got = store.collection("filings").get(include=["documents", "metadatas"])
    chunks = list(zip(got["ids"], got["documents"], got["metadatas"]))
    chunks.sort(key=lambda c: c[0])
    cache = _load_filing_cache()

    uncached = [c for c in chunks if c[0] not in cache]
    todo = uncached[:sample] if sample else uncached
    print(f"filings: {len(chunks)} chunks, {len(cache)} cached, extracting {len(todo)}"
          + (f" (sample of {len(uncached)} uncached)" if sample else ""))

    for start in range(0, len(todo), config.FILING_EXTRACT_BATCH):
        if llm.total_cost() >= config.BUDGET_USD:
            print(f"!! budget guard hit (${llm.total_cost()} >= ${config.BUDGET_USD}) — stopping")
            sys.exit(1)
        batch = todo[start:start + config.FILING_EXTRACT_BATCH]
        items = [(i, batch[i][1]) for i in range(len(batch))]
        result = llm.extract_filing_batch(items)
        with open(config.FILING_EXTRACT_CACHE, "a") as f:
            for i, (cid, _, _) in enumerate(batch):
                f.write(json.dumps({"chunk_id": cid, "risks": result.get(i, [])}) + "\n")
        print(f"  extracted {min(start + len(batch), len(todo))}/{len(todo)}  "
              f"(spend so far: ${llm.total_cost()})")

    # rebuild the table from the cache (idempotent, covers partial runs too)
    cache = _load_filing_cache()
    meta = {cid: m for cid, _, m in chunks}
    store.init_db()
    con = store.db()
    con.execute("DELETE FROM filing_risks")
    n = 0
    for cid, risks in cache.items():
        src = (meta.get(cid) or {}).get("source", "")
        for r in risks:
            con.execute("INSERT INTO filing_risks (chunk_id, source, theme, claim) VALUES (?,?,?,?)",
                        (cid, src, r.get("theme", "").lower().strip(), r["claim"]))
            n += 1
    con.commit()
    con.close()
    print(f"filings: {n} disclosed-risk rows -> SQLite (from {len(cache)} extracted chunks)")
    for t in store.filing_risk_themes(top_n=10):
        print(f"  {t['theme']}: {t['n']} disclosures across {', '.join(t['quarters'])}")


if __name__ == "__main__":
    if "--filing-risks" in sys.argv:
        sample = None
        if "--sample" in sys.argv:
            sample = int(sys.argv[sys.argv.index("--sample") + 1])
        extract_filing_risks(sample=sample)
    else:
        ingest_chat()
        ingest_tesla()
    print(f"\nDONE. total LLM spend: ${llm.total_cost()}")
