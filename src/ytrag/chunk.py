"""Segment[] -> Chunk[]: segments ko overlapping time windows me jodna.

Generic text splitter (RecursiveCharacterTextSplitter) EK lambi string pe kaam
karta hai — timestamps gayab. Yahan timestamp hi product hai, isliye chunking
TIME domain me hoti hai, segment list pe, aur koi segment beech se nahi katta.
"""

import re
from collections import Counter

from ytrag.config import CHUNK_OVERLAP_SECONDS, CHUNK_SECONDS, MIN_CHUNK_WORDS
from ytrag.models import Chunk, Segment, Video

# Sentence boundaries: . ! ? ya newline
_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")


def is_repetitive(text: str, threshold: float = 0.6) -> bool:
    """True agar ek hi sentence chunk ka 60%+ hissa hai.

    Ye Whisper ka loop hallucination hai jo chunk tak bach gaya
    ("Thank you. Thank you. Thank you. ..."). Aise chunks retrieval poison
    hain: har query se thoda-thoda match karte hain aur bolte kuch nahi.
    """
    sentences = [s.strip().lower() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) < 3:
        return False  # 1-2 sentences me "repetition" ka matlab nahi banta
    most_common_count = Counter(sentences).most_common(1)[0][1]
    return most_common_count / len(sentences) >= threshold


def chunk_segments(
    video: Video,
    segments: list[Segment],
    window_seconds: int = CHUNK_SECONDS,
    overlap_seconds: int = CHUNK_OVERLAP_SECONDS,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[Chunk]:
    """Greedy time-window merge, segment boundaries pe overlap ke saath."""
    segments = [s for s in segments if s.text.strip()]
    if not segments:
        return []

    chunks: list[Chunk] = []
    i = 0  # current window ka pehla segment
    while i < len(segments):
        window_start = segments[i].start

        # j ko aage badhao jab tak segment window ke andar khatam ho raha hai
        j = i
        while j < len(segments) and segments[j].end - window_start < window_seconds:
            j += 1
        # Ab j = pehla segment jo window ke BAHAR khatam hota hai (ya len).
        # Usko bhi chunk me le lete hain — segment ko kaatna mana hai, toh
        # chunk thoda 75s se lamba ho sakta hai. Timestamps exact rehte hain.
        last = min(j, len(segments) - 1)
        body_segments = segments[i : last + 1]

        body = " ".join(s.text.strip() for s in body_segments).strip()
        start_sec = int(body_segments[0].start)
        end_sec = int(body_segments[-1].end)

        if len(body.split()) >= min_words and not is_repetitive(body):
            chunks.append(
                Chunk(
                    chunk_id=f"{video.video_id}:{start_sec}",
                    video_id=video.video_id,
                    video_title=video.title,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    # Title prefix: minute 34 ka chunk khud nahi batata ki
                    # topic kya hai. Title batata hai. Sasta trick, bada fayda.
                    text=f"{video.title}\n\n{body}",
                )
            )

        if last >= len(segments) - 1:
            break  # aakhri segment cover ho gaya

        # Overlap: is window ke last ~15 sec me jo segment SHURU hota hai,
        # agla window wahi se shuru karo. Na mile toh seedha last+1 se.
        overlap_start = segments[last].end - overlap_seconds
        next_i = last + 1
        for k in range(i, last + 1):
            if segments[k].start >= overlap_start:
                next_i = k
                break
        # Hamesha aage badho — warna ek 90s ka segment infinite loop bana dega.
        i = max(next_i, i + 1)

    return chunks