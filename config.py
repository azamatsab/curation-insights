"""Central configuration. Everything tunable lives here or in the environment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STORE = ROOT / "store"
STORE.mkdir(exist_ok=True)

# --- data inputs ---
CHAT_JSON = DATA / "interview_synthetic_chat.json"
TESLA_DIR = DATA / "tesla"

# --- persisted stores ---
SQLITE_PATH = STORE / "insights.sqlite"      # structured index (aggregation / ranking)
CHROMA_PATH = str(STORE / "chroma")          # vector index (semantic retrieval)
EXTRACT_CACHE = STORE / "extract_cache.jsonl"  # idempotent LLM-extraction cache
COST_LOG = STORE / "cost_log.jsonl"          # every LLM call's token usage + $ cost

# --- models ---
# Extraction runs over ~1000 short messages, so it's the cost driver -> default to the
# cheapest capable model. Synthesis is a handful of calls. Override via env to use Opus.
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "claude-haiku-4-5")
SYNTH_MODEL = os.getenv("SYNTH_MODEL", "claude-haiku-4-5")

# Messages per extraction call. Batching amortizes the prompt across many messages,
# cutting 1000 calls down to ~50 and keeping cost/latency low.
EXTRACT_BATCH = int(os.getenv("EXTRACT_BATCH", "20"))

# Hard budget guard: extraction aborts if cumulative spend would exceed this.
BUDGET_USD = float(os.getenv("BUDGET_USD", "2.50"))

# Published $/1M tokens (input, output). Used only for local cost estimation/logging.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}

# Tickers the community discusses, with human names for query routing. Extended
# automatically from the data at query time; this is just the seed for name matching.
COMPANY_NAMES = {
    "TSLA": "tesla", "NVDA": "nvidia", "AMZN": "amazon", "COIN": "coinbase",
    "MARA": "marathon digital", "RIOT": "riot platforms", "TSMC": "tsmc taiwan semiconductor",
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "SUI": "sui",
    "URA": "uranium", "ASPI": "asp isotopes", "ONT": "ontology", "YCA": "yellow cake",
    "BUR": "burford capital",
}

# Only Tesla ships primary filings in this task, so only TSLA gets filing augmentation.
FILING_TICKERS = {"TSLA"}
