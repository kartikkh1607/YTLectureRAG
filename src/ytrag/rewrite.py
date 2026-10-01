"""Student ka Hinglish sawaal -> English DSA search query.

Embedding model (MiniLM) English pe train hai. "paas paas wale ghar nahi loot
sakte" uske liye bhasha ka signal zyada hai, topic ka kam — toh woh Hinglish
chunks se match karta hai, House Robber se nahi. Rewrite isko
"house robber adjacent houses maximum sum dynamic programming" bana deta hai.

Trade-off: search ab free/instant nahi — har query pe ek LLM call.
"""

from functools import lru_cache

from ytrag.config import REWRITE_MODEL

REWRITE_PROMPT = """You convert a student's DSA question (Hinglish or English) into a short
English search query for a lecture transcript index.

Rules:
- Output ONLY the query, one line, max 15 words. No quotes, no explanation.
- Use standard English DSA terms (e.g. "adjacent", "subarray", "shortest path").
- If you recognise a well-known problem or algorithm, include its common name
  (e.g. "house robber", "next greater element", "aggressive cows").
- If the question is not about DSA, just translate it to English. Do NOT
  turn it into a DSA question."""


@lru_cache(maxsize=512)
def rewrite_query(question: str) -> str:
    """Rewritten query lautao. Kuch bhi fail ho -> original sawaal (search kabhi na toote).

    lru_cache: same sawaal dobara aaye (eval me, ya CLI me display + search)
    toh LLM call dobara nahi hogi. Pure function hai (same input -> same
    output, temperature=0), isliye cache karna safe hai.
    """
    # Lazy import: answer.py -> index.py -> rewrite.py chain hai. Top pe
    # `from ytrag.answer import ...` likhte toh CIRCULAR IMPORT hota.
    # Function ke andar import call-time pe chalta hai, tab tak sab load ho chuka hota hai.
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
    except Exception:
        return question  # Groq down / quota khatam -> bina rewrite ke search
    # Sanity: khaali ya bahut lamba (model ne explanation likh di) -> original
    if not text or len(text.split()) > 25:
        return question
    return text.splitlines()[0].strip().strip('"')