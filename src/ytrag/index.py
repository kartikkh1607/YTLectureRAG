"""Qdrant: collection banana, chunks upsert karna, stats.

Day14/15 jaisa hi client, par do farak jo scale pe matter karte hain:
  1. collection ke naam me embedding dim (model switch = naya collection)
  2. point ID = uuid5(chunk_id) -> re-run overwrite karta hai, duplicate nahi
(Search M5 me aayega.)
"""

import atexit

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ytrag.config import COLLECTION, QDRANT_API_KEY, QDRANT_PATH, QDRANT_URL, UPSERT_BATCH
from ytrag.embed import get_embedder
from ytrag.models import Chunk

_CLIENT: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Qdrant Cloud agar QDRANT_URL set hai, warna local folder (data/qdrant).

    Local mode = koi account nahi, koi Docker nahi, sirf ek folder.
    Limitation: ek waqt pe ek hi process folder khol sakta hai (file lock).
    """
    global _CLIENT
    if _CLIENT is None:
        if QDRANT_URL:
            _CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            try:
                _CLIENT = QdrantClient(path=str(QDRANT_PATH))
            except RuntimeError as exc:
                # Default error kuch nahi samjhata. Asli wajah batao.
                if "already accessed" in str(exc):
                    raise RuntimeError(
                        "Local Qdrant folder already doosre process ne khola hai — "
                        "shayad `ytrag serve` doosre terminal me chal raha hai. "
                        "Use band karo (Ctrl-C) ya QDRANT_URL set karo."
                    ) from exc
                raise
    return _CLIENT


def _close_client() -> None:
    """Interpreter band hone se PEHLE client close karo.

    Warna Qdrant ka apna cleanup shutdown ke beech chalta hai aur ek
    bekaar sa traceback print karta hai — command sahi chali par crash dikhta hai.
    """
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        except Exception:
            pass
        _CLIENT = None


atexit.register(_close_client)  # Python exit pe ye function khud chalega


def collection_name() -> str:
    """'dsa_lectures_384' — dim naam me, toh MiniLM aur bge-m3 index saath reh sakte hain."""
    return f"{COLLECTION}_{get_embedder().dim}"


def ensure_collection() -> str:
    """Collection nahi hai toh banao. Har baar call karna safe hai (idempotent)."""
    client = get_client()
    name = collection_name()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=get_embedder().dim, distance=Distance.COSINE),
        )
        # Payload index sirf server pe matter karta hai. Local mode bina index
        # ke filter kar leta hai aur index maango toh warning deta hai.
        if QDRANT_URL:
            client.create_payload_index(
                collection_name=name,
                field_name="video_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
    return name


def upsert_chunks(chunks: list[Chunk], batch_size: int = UPSERT_BATCH) -> int:
    """Embed + upsert, batches me. Same chunk_id -> same point_id -> overwrite."""
    if not chunks:
        return 0
    name = ensure_collection()
    client = get_client()
    embedder = get_embedder()

    total = 0
    # Batching kyun: 2933 vectors ek request me = bada payload + ek fail pe sab gaya.
    # 128 ke batches = chhote requests, progress dikhti hai, memory kam.
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        points = [
            PointStruct(id=c.point_id, vector=v, payload=c.to_payload())
            for c, v in zip(batch, vectors)
        ]
        # wait=True -> function tab lautega jab Qdrant ne sach me likh liya
        client.upsert(collection_name=name, points=points, wait=True)
        total += len(points)
    return total


def delete_video(video_id: str) -> None:
    """Ek video ke saare chunks hatao. Chunk size badalne pe zaroori:
    naye start_sec = naye IDs, toh purane points apne aap overwrite NAHI honge."""
    client = get_client()
    client.delete(
        collection_name=ensure_collection(),
        points_selector=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        ),
        wait=True,
    )


def indexed_video_ids() -> set[str]:
    """Kaunse videos already index me hain — re-run inhe skip kar sakta hai."""
    client = get_client()
    name = collection_name()
    if not client.collection_exists(name):
        return set()

    found: set[str] = set()
    offset = None
    # scroll = pagination. Ek baar me 1000 points, `offset` agla page batata hai,
    # None aaye matlab khatam. with_vectors=False -> sirf payload, tez.
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=["video_id"],
            with_vectors=False,
        )
        for p in points:
            if p.payload and p.payload.get("video_id"):
                found.add(p.payload["video_id"])
        if offset is None:
            break
    return found


def count_points() -> int:
    client = get_client()
    name = collection_name()
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name, exact=True).count