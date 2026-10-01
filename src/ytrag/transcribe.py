"""Transcript cache: JSON files in data/transcripts/.

Ye project ki sabse important file hai. Ek transcript = ghanton ka GPU time.
Galti se re-transcribe karna ya aadhi-likhi file pe bharosa karna, dono mehenge hain.

M2a (abhi): cache padhna aur atomically likhna.
M2b (baad me): yt-dlp + faster-whisper se naya transcript banana.
"""

import json
import os
from pathlib import Path

from ytrag.config import TRANSCRIPT_DIR
from ytrag.models import Segment, Video


def transcript_path(video_id: str, directory: Path | None = None) -> Path:
    # directory param testing ke liye hai — check script precious folder ko
    # chhue bina ek temp folder me likh ke test kar sakti hai.
    return (directory or TRANSCRIPT_DIR) / f"{video_id}.json"


def load_transcript(video_id: str, directory: Path | None = None) -> dict | None:
    """Cached transcript dict lautao, ya None agar file nahi hai / kharab hai.

    None ka matlab hai "cache miss" — caller dobara transcribe karega.
    Crash karne se better hai, kyunki ek kharab file poora ingest na roke.
    """
    path = transcript_path(video_id, directory)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    # Valid JSON hai par galat shape? Woh bhi cache miss.
    if not isinstance(data, dict) or "segments" not in data:
        return None
    return data


def cached_video_ids(directory: Path | None = None) -> list[str]:
    """Jitne transcripts disk pe hain, unke video_ids (sorted, taaki order fixed rahe)."""
    folder = directory or TRANSCRIPT_DIR
    if not folder.exists():
        return []
    # p.stem = filename bina extension ke: "mMKZmo3iyyM.json" -> "mMKZmo3iyyM"
    return sorted(p.stem for p in folder.glob("*.json"))


def segments_from_transcript(data: dict) -> list[Segment]:
    return [
        Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"])
        for s in data.get("segments", [])
    ]


def video_from_transcript(data: dict) -> Video:
    """Transcript me title/duration already saved hai — reindex ke waqt
    YouTube se dobara poochne ki zarurat nahi. (Original me ye helper nahi tha.)"""
    return Video(
        video_id=data["video_id"],
        title=data.get("title") or data["video_id"],
        duration=int(data.get("duration") or 0),
    )


def write_transcript(
    video: Video,
    segments: list[Segment],
    language: str,
    model_name: str,
    directory: Path | None = None,
) -> Path:
    """ATOMIC write: pehle .tmp file me likho, phir os.replace se rename.

    Seedha final file me likhte waqt Ctrl-C / power cut aaya toh aadhi JSON
    bachegi. Agar woh aadhi file bhi parse ho gayi (jaise segments list
    beech me kati par valid), toh cache use hamesha "complete" maanega.

    os.replace same drive pe atomic hai: file ya toh poori nayi hogi, ya
    purani hi rahegi. Beech ki koi state exist nahi karti.
    """
    folder = directory or TRANSCRIPT_DIR
    folder.mkdir(parents=True, exist_ok=True)  # writer khud folder banata hai

    payload = {
        "video_id": video.video_id,
        "title": video.title,
        "language": language,
        "model": model_name,
        "duration": video.duration,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }

    final = transcript_path(video.video_id, folder)
    tmp = final.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        # ensure_ascii=False -> Hindi/Devanagari text \u0915 jaisa escape nahi hoga
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, final)
    return final