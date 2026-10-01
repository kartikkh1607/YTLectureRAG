"""M1 sanity check: config paths, ek real transcript, aur Chunk ke properties."""

import json

from ytrag.config import PROJECT_ROOT, TRANSCRIPT_DIR
from ytrag.models import Chunk, Segment, Video, format_timestamp

print("Project root  :", PROJECT_ROOT)
print("Transcripts   :", TRANSCRIPT_DIR, "| exists:", TRANSCRIPT_DIR.exists())

# 1. Ek real transcript load karke Video + Segments banao
path = TRANSCRIPT_DIR / "mMKZmo3iyyM.json"
data = json.loads(path.read_text(encoding="utf-8"))

video = Video(video_id=data["video_id"], title=data["title"], duration=data["duration"])
segments = [Segment(s["start"], s["end"], s["text"]) for s in data["segments"]]

print("\nVideo         :", video.title)
print("URL           :", video.url)
print("Duration      :", format_timestamp(video.duration))
print("Segments      :", len(segments))
print("Segment[1]    :", f"{segments[1].start:.1f}s -> {segments[1].end:.1f}s |", segments[1].text[:70], "...")

# 2. Haath se ek fake chunk banao (asli chunking M3 me)
chunk = Chunk(
    chunk_id=f"{video.video_id}:735",
    video_id=video.video_id,
    video_title=video.title,
    start_sec=735,
    end_sec=810,
    text="dummy text",
)
print("\nchunk_id      :", chunk.chunk_id)
print("point_id      :", chunk.point_id)
print("same again?   :", chunk.point_id == Chunk(**{**chunk.__dict__}).point_id)
print("link_sec      :", chunk.link_sec, "(735 - 5 rewind)")
print("timestamp     :", chunk.timestamp)
print("url           :", chunk.url)

# 3. Payload round-trip: to_payload -> from_payload -> wahi chunk wapas?
print("round-trip ok :", Chunk.from_payload(chunk.to_payload()) == chunk)