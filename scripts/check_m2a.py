"""M2a check: poora cache scan karo + atomic write ka round-trip test."""

import shutil

from ytrag.config import DATA_DIR
from ytrag.models import Segment, format_timestamp
from ytrag.transcribe import (
    cached_video_ids,
    load_transcript,
    segments_from_transcript,
    video_from_transcript,
    write_transcript,
)

# ---------- 1. Poora cache scan ----------
ids = cached_video_ids()
total_sec = 0.0
total_segments = 0
bad = []

for vid in ids:
    data = load_transcript(vid)
    if data is None:
        bad.append(vid)
        continue
    segs = segments_from_transcript(data)
    total_segments += len(segs)
    if segs:
        total_sec += segs[-1].end  # last segment ka end ~ video length

print(f"Transcripts      : {len(ids)}")
print(f"Corrupt/invalid  : {len(bad)} {bad}")
print(f"Total speech     : {total_sec / 3600:.1f} hours")
print(f"Total segments   : {total_segments}")
print(f"Avg segment      : {total_sec / total_segments:.1f} sec")

# Sabse lamba aur sabse chhota lecture
lengths = []
for vid in ids:
    data = load_transcript(vid)
    if data and data["segments"]:
        lengths.append((data["segments"][-1]["end"], video_from_transcript(data).title))
lengths.sort()
print(f"Shortest         : {format_timestamp(lengths[0][0])}  {lengths[0][1][:60]}")
print(f"Longest          : {format_timestamp(lengths[-1][0])}  {lengths[-1][1][:60]}")

# ---------- 2. Atomic write round-trip (temp folder, precious data safe) ----------
tmp_dir = DATA_DIR / "_check_tmp"
original = load_transcript(ids[0])
video = video_from_transcript(original)
segments = segments_from_transcript(original)

path = write_transcript(video, segments, original["language"], original["model"], directory=tmp_dir)
reloaded = load_transcript(video.video_id, directory=tmp_dir)

print(f"\nWrote            : {path.name}")
print(f"Leftover .tmp?   : {list(tmp_dir.glob('*.tmp'))}")
print(f"Round-trip equal : {segments_from_transcript(reloaded) == segments}")

# ---------- 3. Corrupt file -> None, crash nahi ----------
broken = tmp_dir / "BROKEN.json"
broken.write_text('{"video_id": "BROKEN", "segments": [{"start": 0', encoding="utf-8")
print(f"Truncated JSON   : {load_transcript('BROKEN', directory=tmp_dir)}")

shutil.rmtree(tmp_dir)
print(f"Cleanup          : {not tmp_dir.exists()}")