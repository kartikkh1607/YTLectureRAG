"""Retrieve -> grounded answer + citations.

Sabse important hissa: jab retrieval khaali aaye, toh refusal lautao
BINA LLM call kiye.

Model ko DSA pehle se aata hai. Junk context do toh woh apni training se
answer kar dega aur TUMHARE timestamps chipka dega — student click karega
aur video me kuch aur chal raha hoga. "Cover nahi hua" bolna usse behtar hai.
"""

import re
from functools import lru_cache

from groq import Groq

from ytrag.config import (
    CONFIDENT_DISTANCE,
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    QUERY_REWRITE,
    REFUSAL,
    TOP_K,
)
from ytrag.index import search, title_overlap
from ytrag.models import Chunk

SYSTEM_PROMPT = f"""You are answering using ONLY the transcript excerpts below, which come from
Pratyush's DSA lectures. The transcripts are auto-generated and may contain
minor errors — read past obvious mis-transcriptions of technical terms.

Rules:
- Answer only from the excerpts. If they don't cover it, say exactly:
  "{REFUSAL}"
- Cite with [1], [2] inline, using the excerpt numbers given below.
- Always answer in English, even if the question is in Hinglish.
- 4-6 sentences max.
- Never invent a timestamp or a lecture name."""

# [1], [12] jaise citations pakadne ke liye. (\d+) = group(1) = number.
_CITATION_RE = re.compile(r"\[(\d+)\]")

_CLIENT: Groq | None = None


def get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY set nahi hai. .env check karo.")
        _CLIENT = Groq(api_key=GROQ_API_KEY)
    return _CLIENT


def _chat(system: str, user: str) -> str:
    """Ek completion. Provider badalna ho toh sirf ye function badlega."""
    response = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Grounded QA = creativity nahi chahiye. 0 -> har baar sabse likely
        # token, same sawaal pe (lagbhag) same answer. Eval ke liye bhi zaroori.
        temperature=0,
        max_completion_tokens=GROQ_MAX_TOKENS,
        # gpt-oss ka thinking budget. Excerpts padh ke 5 line likhna = halka kaam.
        reasoning_effort="low",
    )
    content = response.choices[0].message.content
    # content None ho sakta hai (jaise token budget reasoning me hi khatam).
    # Chupchaap "" lautane se better hai saaf error.
    if not content:
        finish = response.choices[0].finish_reason
        raise RuntimeError(f"LLM ne khaali answer diya (finish_reason={finish}).")
    return content.strip()


def build_context(chunks: list[Chunk]) -> str:
    """Excerpts ko numbered blocks me: [1] "title" @ 12:10 \\n text ..."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f'[{i}] "{chunk.video_title}" @ {chunk.timestamp}\n{chunk.text}')
    return "\n\n".join(blocks)


@lru_cache(maxsize=256)
def video_duration(video_id: str) -> int:
    """Lecture ki length (sec) — UI timeline ke liye. Transcript ka last segment end.

    lru_cache: har search pe 6 JSON files padhna bekaar; video ki length badalti nahi.
    """
    from ytrag.transcribe import load_transcript

    data = load_transcript(video_id)
    if not data or not data.get("segments"):
        return 0
    return int(data["segments"][-1]["end"])


def _citation(chunk: Chunk, distance: float) -> dict:
    return {
        "title": chunk.video_title,
        "timestamp": chunk.timestamp,
        "url": chunk.url,
        "start_sec": chunk.link_sec,
        "end_sec": chunk.end_sec,
        "duration": video_duration(chunk.video_id),
        "video_id": chunk.video_id,
        "distance": round(distance, 4),
    }


def _renumber(text: str, hits: list[tuple[Chunk, float]]) -> tuple[str, list[dict]]:
    """Sirf woh citations rakho jo model ne USE kiye, aur unhe 1..N renumber karo.

    Model ne [2] aur [5] use kiya -> answer me [1] aur [2] dikhega, aur neeche
    sirf 2 links. Warna student 6 links dekhega jinme se 4 bekaar — aur phir
    kisi pe bharosa nahi karega.
    """
    order: list[int] = []  # pehli baar dikhne ke order me, bina duplicate
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        # 1 <= idx <= len(hits): model ne [9] likh diya jabki 6 excerpts the?
        # Woh hallucinated citation hai — ignore.
        if 1 <= idx <= len(hits) and idx not in order:
            order.append(idx)

    if not order:
        return text, []

    remap = {old: new for new, old in enumerate(order, start=1)}  # {2: 1, 5: 2}
    rewritten = _CITATION_RE.sub(
        # valid number -> naya number; invalid ([9]) -> hata do
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else "",
        text,
    )
    citations = [_citation(*hits[old - 1]) for old in order]
    return rewritten, citations


def answer(question: str, top_k: int = TOP_K, video_id: str | None = None) -> dict:
    """-> {"answer", "citations", "grounded", "retrieved"}"""
    question = question.strip()
    if not question:
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": 0}

    hits = search(question, top_k=top_k, video_id=video_id)

    # GUARD 1: cutoff ke baad kuch nahi bacha -> refusal, LLM call hi nahi.
    # Sasta (0 tokens), tez, aur hallucinate karna impossible.
    if not hits:
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": 0}

    chunks = [chunk for chunk, _ in hits]
    user_prompt = f"EXCERPTS\n{build_context(chunks)}\n\nQUESTION: {question}"
    text = _chat(SYSTEM_PROMPT, user_prompt)

    # GUARD 2: model ne excerpts padh ke khud bola "cover nahi hua" -> maano.
    # (Cutoff 0.6 coarse hai; semantic judgement yahin hota hai.)
    if REFUSAL.lower() in text.lower():
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": len(hits)}

    text, citations = _renumber(text, hits)

    # GUARD 3: ek bhi citation nahi = model apni knowledge se bol raha hai.
    # Answer dikhao, par grounded=False aur koi link nahi.
    return {
        "answer": text,
        "citations": citations,
        "grounded": bool(citations),
        "retrieved": len(hits),
    }


def search_only(question: str, top_k: int = TOP_K, video_id: str | None = None) -> dict:
    """UI ka main path: ranked timestamps, answer-LLM ke bina.

    Student ko paraphrase nahi, lecture ka woh second chahiye. Ye path
    hallucinate kar hi nahi sakta — kuch generate nahi hota.
    (Rewrite ON hai toh ek chhoti LLM call query ke liye lagti hai, bas.)

    `confident` sirf advisory hai, gate nahi: weak results bhi dikhte hain,
    bas UI unhe "shayad" bol ke dikhata hai.
    """
    question = question.strip()
    if not question:
        return {"query": question, "effective_query": question, "confident": False, "results": []}

    effective = question
    if QUERY_REWRITE:
        from ytrag.rewrite import rewrite_query

        effective = rewrite_query(question)  # cached; search() dobara call karega -> free

    hits = search(question, top_k=top_k, video_id=video_id)

    # Do signals: title me topic hai, YA distance kaafi kam hai.
    # Sirf distance pe bharosa nahi — M5 me dekha, off-topic aur genuine overlap karte hain.
    confident = bool(hits) and (
        title_overlap(effective, hits[0][0].video_title) > 0
        or hits[0][1] <= CONFIDENT_DISTANCE
    )
    return {
        "query": question,
        "effective_query": effective,
        "confident": confident,
        "results": [
            {
                "title": c.video_title,
                "video_id": c.video_id,
                "start_sec": c.link_sec,
                "end_sec": c.end_sec,
                "duration": video_duration(c.video_id),  # UI timeline: clip lecture me kahan hai
                "timestamp": c.timestamp,
                "url": c.url,
                "distance": round(d, 4),
                # chunk.text = "title\n\nbody" -> sirf body ka shuru
                "preview": c.text.split("\n\n", 1)[-1][:220].strip(),
            }
            for c, d in hits
        ],
    }