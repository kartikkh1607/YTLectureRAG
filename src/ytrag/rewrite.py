"""Student ka Hinglish sawaal -> English DSA search query.

Embedding model (MiniLM) English pe train hai. "paas paas wale ghar nahi loot
sakte" uske liye bhasha ka signal zyada hai, topic ka kam — toh woh Hinglish
chunks se match karta hai, House Robber se nahi. Rewrite isko
"house robber adjacent houses maximum sum dynamic programming" bana deta hai.

Trade-off: search ab free/instant nahi — har NAYE sawaal pe ek LLM call.

Disk cache (data/rewrite_cache.json) kyun:
  1. Eval reproducible: temperature=0 bhi 100% deterministic nahi. Same sawaal
     ka rewrite run-to-run badla toh downstream change (stemmer, boost) ka
     asar noise me chhup jaata hai. Cache = upstream FREEZE.
  2. Production: wahi sawaal dobara aaya -> 0 tokens, 0 latency.
"""

import hashlib
import json
import os
import sys
import threading

from ytrag.config import REWRITE_MODEL, WRITABLE_DIR

REWRITE_PROMPT = """You convert a student's DSA question (Hinglish or English) into a short
English search query for a lecture transcript index.

Rules:
- Output ONLY the query, one line, max 15 words. No quotes, no explanation.
- Use standard English DSA terms (e.g. "adjacent", "subarray", "shortest path").
- If you recognise a well-known problem or algorithm, include its common name
  (e.g. "house robber", "next greater element", "aggressive cows").
- If the question is not about DSA, just translate it to English. Do NOT
  turn it into a DSA question."""

# WRITABLE_DIR: local pe data/, Vercel pe /tmp/ytrag (wahan baaki sab read-only).
CACHE_PATH = WRITABLE_DIR / "rewrite_cache.json"

# Cache key me model + prompt ka hash. Prompt ya model badla -> naya hash ->
# purane rewrites apne aap ignore. (HireMeAI wala lesson: cache key me har
# woh cheez daalo jo output badal sakti hai.)
_VERSION = hashlib.sha256((REWRITE_MODEL + "\n" + REWRITE_PROMPT).encode()).hexdigest()[:12]

_cache: dict[str, str] | None = None
# FastAPI sync endpoints threadpool me chalte hain -> do requests ek saath
# cache badal sakti hain. json.dumps chal raha ho aur doosra thread dict me
# key daale -> "dictionary changed size during iteration". Lock se ek waqt
# pe ek hi thread cache chhuega.
_lock = threading.Lock()


def _load() -> dict[str, str]:
    """Pehli call pe disk se padho, phir memory me rakho."""
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _cache = {}  # file nahi / kharab -> khaali se shuru, crash nahi
    return _cache


def _save(cache: dict[str, str]) -> None:
    """Atomic write — transcribe.py wala same pattern (tmp + os.replace)."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, CACHE_PATH)


def _call_llm(question: str) -> str | None:
    """LLM se rewrite. Fail ya bekaar output -> None."""
    # Lazy import: answer.py -> index.py -> rewrite.py chain hai. Top pe
    # `from ytrag.answer import ...` likhte toh CIRCULAR IMPORT hota.
    from ytrag.answer import get_client

    try:
        response = get_client().chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_completion_tokens=1024,  # reasoning model: thinking bhi isi me
            reasoning_effort="low",
        )
        text = (response.choices[0].message.content or "").strip()
        finish = response.choices[0].finish_reason
    except Exception as exc:
        # Fallback chupchaap tha -> production me rewrite band hua aur kisi ko pata
        # nahi chala (M9 bug). stderr = Vercel runtime logs. Sirf exception ka
        # type + message (200 chars) — API key ya headers kabhi log mat karo.
        print(
            f"[rewrite] LLM call failed: {type(exc).__name__}: {str(exc)[:200]}",
            file=sys.stderr,
            flush=True,
        )
        return None
    # Sanity: khaali ya bahut lamba (model ne explanation likh di) -> bekaar
    if not text or len(text.split()) > 25:
        # finish_reason="length" = reasoning ne token budget kha liya, answer aaya hi nahi.
        print(
            f"[rewrite] output rejected: {'empty' if not text else 'too long'} "
            f"(words={len(text.split())}, finish_reason={finish})",
            file=sys.stderr,
            flush=True,
        )
        return None
    return text.splitlines()[0].strip().strip('"')


def rewrite_query(question: str) -> str:
    """Cached rewrite lautao; miss pe LLM; LLM fail -> original sawaal."""
    question = question.strip()
    key = f"{_VERSION}:{question}"
    with _lock:
        cache = _load()
        if key in cache:
            return cache[key]

    # LLM call lock ke BAHAR — warna ek slow call baaki sab requests ko rok degi.
    result = _call_llm(question)
    if result is None:
        # FAILURE CACHE MAT KARO. Groq 2 min down tha aur humne "original
        # sawaal" cache kar diya -> woh sawaal HAMESHA bina rewrite ke chalega.
        return question

    with _lock:
        cache[key] = result
        _save(cache)
    return result