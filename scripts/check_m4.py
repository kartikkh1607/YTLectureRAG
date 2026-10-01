"""M4 check: saare transcripts -> chunks -> embed -> Qdrant. Phir idempotency test."""

import time

from ytrag.chunk import chunk_segments
from ytrag.embed import get_embedder
from ytrag.index import collection_name, count_points, indexed_video_ids, upsert_chunks
from ytrag.transcribe import (
    cached_video_ids,
    load_transcript,
    segments_from_transcript,
    video_from_transcript,
)

t0 = time.perf_counter()
embedder = get_embedder()
print(f"Embedder   : {embedder.name} ({embedder.dim}-dim), loaded in {time.perf_counter() - t0:.1f}s")
print(f"Collection : {collection_name()}")
print(f"Before     : {count_points()} points")

# ---------- 1. Index everything ----------
ids = cached_video_ids()
t0 = time.perf_counter()
total = 0
for n, vid in enumerate(ids, start=1):
    data = load_transcript(vid)
    chunks = chunk_segments(video_from_transcript(data), segments_from_transcript(data))
    total += upsert_chunks(chunks)
    if n % 20 == 0 or n == len(ids):
        print(f"  {n:>3}/{len(ids)} videos | {total} chunks | {time.perf_counter() - t0:.0f}s")

print(f"After      : {count_points()} points")
print(f"Videos     : {len(indexed_video_ids())} indexed")

# ---------- 2. Idempotency: same video dobara upsert -> count same rahe ----------
data = load_transcript(ids[0])
chunks = chunk_segments(video_from_transcript(data), segments_from_transcript(data))
before = count_points()
upsert_chunks(chunks)
print(f"\nRe-upsert {len(chunks)} chunks of {ids[0]}: {before} -> {count_points()} points")