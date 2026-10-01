"""Teen data structures jo poori pipeline me ghoomte hain: Video, Segment, Chunk."""

import uuid
from dataclasses import dataclass

from ytrag.config import LINK_REWIND_SECONDS

# Qdrant point ID sirf unsigned int ya UUID accept karta hai.
# Hamara readable chunk_id ("mMKZmo3iyyM:735") string hai, toh usko
# uuid5 se hash karke UUID banate hain.
#
# uuid5(namespace, text) DETERMINISTIC hai: same input -> hamesha same UUID.
# Isi se re-ingest "overwrite" banta hai, "duplicate" nahi.
#
# Namespace koi bhi fixed UUID ho sakta hai — bas index banne ke baad
# kabhi badalna nahi, warna saare IDs badal jaayenge aur sab duplicate hoga.
_NAMESPACE = uuid.UUID("35bc4ba9-689e-4264-92a1-4db7b0ec39e2")

def format_timestamp(seconds: int) -> str:
    """735 -> '12:15', 4335 -> '1:12:15'."""
    seconds = max(0, int(seconds))
    # divmod(a, b) -> (a // b, a % b) ek saath
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class Video:
    video_id: str
    title: str
    duration: int = 0  # seconds

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass
class Segment:
    """Whisper ka raw output: ek utterance, apne start/end ke saath."""

    start: float
    end: float
    text: str


@dataclass
class Chunk:
    """Jo actually embed hota hai: kai segments jud ke ek time window."""

    chunk_id: str  # f"{video_id}:{start_sec}" — re-ingest pe bhi same rehta hai
    video_id: str
    video_title: str
    start_sec: int
    end_sec: int
    text: str

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_NAMESPACE, self.chunk_id))

    @property
    def link_sec(self) -> int:
        """Citation link kahan land kare — chunk se thoda pehle (0 se neeche nahi)."""
        return max(0, self.start_sec - LINK_REWIND_SECONDS)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}&t={self.link_sec}s"

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.link_sec)

    def to_payload(self) -> dict:
        """Qdrant payload. Flat rakho, sirf primitives — no None, no nested dict."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "Chunk":
        """Qdrant se wapas aaya dict -> Chunk object. to_payload ka ulta."""
        return cls(
            chunk_id=payload["chunk_id"],
            video_id=payload["video_id"],
            video_title=payload["video_title"],
            start_sec=int(payload["start_sec"]),
            end_sec=int(payload["end_sec"]),
            text=payload["text"],
        )