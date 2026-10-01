"""M3 check: pehle toy data pe algorithm samjho, phir real transcripts pe chalao."""

from ytrag.chunk import chunk_segments, is_repetitive
from ytrag.models import Segment, Video, format_timestamp
from ytrag.transcribe import (
    cached_video_ids,
    load_transcript,
    segments_from_transcript,
    video_from_transcript,
)

# ---------- 1. Toy example: chhote segments, haath se trace karne layak ----------
toy_video = Video("TOY", "Toy Lecture")
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa "  # 10 words
toy = [Segment(t, t + 10, f"seg{t // 10} " + words) for t in range(0, 200, 10)]  # 20 x 10s

print("=== Toy: 20 segments x 10s, window=40s, overlap=15s ===")
for c in chunk_segments(toy_video, toy, window_seconds=40, overlap_seconds=15):
    first = c.text.split("\n\n")[1].split()[0]
    print(f"  {c.start_sec:>3}s -> {c.end_sec:>3}s  ({c.end_sec - c.start_sec}s)  starts with {first}")

# ---------- 2. is_repetitive ----------
print("\n=== is_repetitive ===")
print("  loop  :", is_repetitive("Thank you. Thank you. Thank you. Thank you. Okay."))
print("  normal:", is_repetitive("Stack is LIFO. Push adds on top. Pop removes from top."))

# ---------- 3. Ek real video ----------
data = load_transcript("mMKZmo3iyyM")
video = video_from_transcript(data)
segments = segments_from_transcript(data)
chunks = chunk_segments(video, segments)

print(f"\n=== {video.title[:50]} ===")
print(f"  segments -> chunks : {len(segments)} -> {len(chunks)}")
for c in chunks[:4]:
    print(f"  [{format_timestamp(c.start_sec)} - {format_timestamp(c.end_sec)}] {c.end_sec - c.start_sec}s  {c.chunk_id}")
print("  chunk[2] text preview:")
print("   ", chunks[2].text[:220].replace("\n", " | "), "...")

# ---------- 4. Poora corpus ----------
all_chunks = []
for vid in cached_video_ids():
    d = load_transcript(vid)
    all_chunks += chunk_segments(video_from_transcript(d), segments_from_transcript(d))

durations = [c.end_sec - c.start_sec for c in all_chunks]
ids = [c.chunk_id for c in all_chunks]

# Overlap kitni baar sach me hua? (agla chunk pichle ke khatam hone se pehle shuru)
pairs = overlapping = 0
for a, b in zip(all_chunks, all_chunks[1:]):
    if a.video_id == b.video_id:
        pairs += 1
        overlapping += b.start_sec < a.end_sec

print("\n=== Full corpus ===")
print(f"  total chunks       : {len(all_chunks)}")
print(f"  unique chunk_ids   : {len(set(ids))}")
print(f"  avg / min / max    : {sum(durations) / len(durations):.0f}s / {min(durations)}s / {max(durations)}s")
print(f"  pairs that overlap : {overlapping}/{pairs} ({100 * overlapping / pairs:.0f}%)")