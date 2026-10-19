"""FastAPI: /search (main), /ask (LLM explanation), /health, /stats + UI.

Run: uv run ytrag serve   ->  http://127.0.0.1:8000
"""

import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from groq import APIStatusError, RateLimitError
from pydantic import BaseModel, Field

from ytrag import config
from ytrag.answer import answer, search_only

STATIC_DIR = Path(__file__).resolve().parent / "static"


# Startup ka haal: component -> None (theek) ya error message. Khaali = lifespan
# abhi chala hi nahi.
_startup: dict[str, str | None] = {}


def _load(component: str, fn) -> None:
    """fn chalao; fail ho toh poora traceback stderr (Vercel logs) pe, app zinda rahe.

    Pehle yahan ek bhi error = poora app crash, har route 500 — /health bhi.
    Ab sirf search/ask 503 denge, aur /health batayega kya toota.
    """
    try:
        fn()
        _startup[component] = None
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _startup[component] = f"{type(exc).__name__}: {exc}"[:500]
        print(f"[startup] {component} failed to load: {_startup[component]}", file=sys.stderr, flush=True)


def _require(*components: str) -> None:
    """Jo component chahiye woh load nahi hua -> 503, saaf message ke saath."""
    missing = [c for c in components if c not in _startup or _startup[c]]
    if missing:
        raise HTTPException(
            503,
            f"Search is temporarily unavailable: the server could not load the "
            f"{' and '.join(missing)}. Check /health for details.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server start pe hi model + Qdrant load karo.

    Warna pehla student ~10 sec spinner dekhega (lazy load). Wait terminal
    me ho, jahan expected hai — user ke saamne nahi.
    """
    from ytrag.embed import get_embedder
    from ytrag.index import count_points, get_client

    print("Loading embedding model...", flush=True)
    _load("embedder", get_embedder)
    _load("qdrant", lambda: (get_client(), count_points()))
    if not any(_startup.values()):
        emb = get_embedder()
        print(f"Ready: {emb.name} ({emb.dim}-dim), {count_points()} chunks", flush=True)
    yield  # yield se pehle = startup, baad me = shutdown


app = FastAPI(title="YT Lecture RAG", lifespan=lifespan)


class Query(BaseModel):
    # Pydantic validation: khaali ya 300+ chars ka sawaal -> 422, function tak pahunchega hi nahi.
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


# ------------------------------------------------------------------
# Rate limiting: sliding window, per-IP + global
# ------------------------------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)
_global: deque = deque()
_rl_lock = threading.Lock()


def _allow(window: deque, limit: int, now: float) -> bool:
    """Window se purane timestamps nikaalo; limit se kam hai toh allow + record."""
    while window and now - window[0] > config.RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= limit:
        return False
    window.append(now)
    return True


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()  # monotonic: system clock badle toh bhi peeche nahi jaata
    with _rl_lock:
        if not _allow(_global, config.RATE_LIMIT_GLOBAL, now):
            raise HTTPException(429, "The server is busy right now. Try again in a minute.")
        if not _allow(_hits[ip], config.RATE_LIMIT_PER_IP, now):
            raise HTTPException(
                429,
                f"Too many questions. The limit is {config.RATE_LIMIT_PER_IP} every "
                f"{config.RATE_LIMIT_WINDOW_SECONDS} seconds.",
            )
        # Original me ye nahi tha: har naya IP dict me hamesha ke liye reh jaata
        # (memory leak). 1000 se zyada entries -> khaali windows saaf karo.
        if len(_hits) > 1000:
            for key in [k for k, w in _hits.items() if not w]:
                del _hits[key]


def _llm_errors(fn, *args, **kwargs):
    """Groq ke errors ko sahi HTTP codes me badlo — student ko 500 se kuch samajh nahi aata."""
    try:
        return fn(*args, **kwargs)
    except RateLimitError:
        raise HTTPException(429, "The AI provider's daily limit is used up. Search still works; try Explain again tomorrow.")
    except APIStatusError as exc:
        raise HTTPException(502, f"The AI provider returned an error ({exc.status_code}). Try again.")
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health")
def health():
    components = {c: err or "ok" for c, err in _startup.items()}
    if _startup and not any(_startup.values()):
        return {"status": "ok", "components": components}
    # 503 taaki uptime checks bhi pakdein; body me exact error.
    status = "degraded" if _startup else "starting"
    return JSONResponse(status_code=503, content={"status": status, "components": components})


@lru_cache(maxsize=1)
def _library() -> dict:
    """Kitne lectures, kitne ghante — header me dikhane ke liye. Ek baar compute."""
    from ytrag.answer import video_duration
    from ytrag.transcribe import cached_video_ids

    ids = cached_video_ids()
    return {"lectures": len(ids), "hours": round(sum(video_duration(v) for v in ids) / 3600, 1)}


@app.get("/stats")
def stats():
    from ytrag.index import collection_name, count_points

    _require("qdrant")
    return {
        "collection": collection_name(),
        "chunks": count_points(),
        "rewrite": config.QUERY_REWRITE,
        # Sirf boolean — "key Vercel tak pahunchi ya nahi" debug karne ke liye. Value kabhi nahi.
        "groq_key_set": bool(config.GROQ_API_KEY),
        **_library(),
    }


@app.post("/search")
def search_route(q: Query, request: Request):
    # Original me /search rate-limited nahi tha kyunki free tha. Ab rewrite ki
    # LLM call lagti hai -> tokens lagte hain -> limit zaroori.
    _require("embedder", "qdrant")  # rate limit se pehle: toota server quota na khaaye
    rate_limit(request)
    return _llm_errors(search_only, q.question, top_k=q.top_k)


@app.post("/ask")
def ask_route(q: Query, request: Request):
    _require("embedder", "qdrant")
    rate_limit(request)
    return _llm_errors(answer, q.question, top_k=q.top_k)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")