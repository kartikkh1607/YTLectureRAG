"""Text -> vector. Model ek baar load hota hai, poore process me reuse."""

from typing import Protocol

import numpy as np

from ytrag.config import (
    EMBED_BATCH,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
    EMBED_THREADS,
    MODEL_CACHE_DIR,
)


class Embedder(Protocol):
    """Contract: jo bhi embedder ho, uske paas ye 4 cheezein honi chahiye.

    Protocol = "structural typing". Kisi class ko isse inherit nahi karna
    padta — bas ye methods ho, toh woh Embedder hai. Kal OpenAI/Cohere
    embedder lagana ho toh baaki code (index.py) ko pata bhi nahi chalega.
    """

    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vector: np.ndarray) -> list[float]:
    """Length 1 pe normalize -> cosine similarity = dot product.

    Qdrant COSINE bhi andar normalize karta hai, par yahan karne se vectors
    kahin bhi (numpy, export) consistent rehte hain. fastembed MiniLM ko
    already normalize karta hai — par ye uski andar ki detail hai, contract
    nahi. Model badla toh bhi guarantee yahin se aayegi.
    """
    norm = float(np.linalg.norm(vector))
    if norm > 0:  # zero vector (khaali text?) ko 0 se divide mat karo
        vector = vector / norm
    # numpy array -> plain Python list (Qdrant client ko JSON-able chahiye)
    return vector.tolist()


class FastEmbedEmbedder:
    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        batch_size: int = EMBED_BATCH,
        threads: int | None = EMBED_THREADS,
    ):
        # Import yahan andar — onnxruntime load hone me ~1 sec lagta hai.
        # Jo commands embedder use nahi karti (jaise `ytrag --help`), woh tez rahengi.
        from fastembed import TextEmbedding

        # dim fastembed ki metadata se — model load kiye bina pata chal jaata hai.
        meta = next(
            (m for m in TextEmbedding.list_supported_models() if m["model"] == model_name),
            None,
        )
        if meta is None:
            raise RuntimeError(
                f"fastembed '{model_name}' support nahi karta. "
                "TextEmbedding.list_supported_models() me exact naam dekho "
                "(jaise 'sentence-transformers/all-MiniLM-L6-v2')."
            )

        self.name = model_name
        self.dim = int(meta["dim"])
        self.batch_size = batch_size
        # Download yahan hoga. Vercel pe default HF cache (~/.cache) read-only hai.
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(MODEL_CACHE_DIR),  # pehli baar HF se ~90 MB download
            threads=threads,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # .embed() generator deta hai (lazy) — list me badlo, har vector normalize.
        return [_normalize(v) for v in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self.model.embed([EMBED_QUERY_PREFIX + text])))
        return _normalize(vector)


# Module-level singleton: model load mehenga hai, process me ek hi baar.
_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = FastEmbedEmbedder()
    return _EMBEDDER