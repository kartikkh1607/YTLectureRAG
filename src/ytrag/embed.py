"""Text -> vector. Model ek baar load hota hai, poore process me reuse."""

from typing import Protocol

from ytrag.config import EMBED_BATCH, EMBED_MODEL, EMBED_QUERY_PREFIX


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


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH):
        # Import yahan andar — sentence_transformers torch kheenchta hai (~2-5 sec).
        # Jo commands embedder use nahi karti (jaise stats), woh tez rahengi.
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)  # pehli baar HF se download
        # sentence-transformers v6 me method ka naam badla — dono versions pe chale
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            # Length 1 pe normalize -> cosine similarity = dot product.
            # Qdrant COSINE bhi andar normalize karta hai, par yahan karne se
            # vectors kahin bhi (numpy, export) consistent rehte hain.
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # numpy array -> plain Python list (Qdrant client ko JSON-able chahiye)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            EMBED_QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


# Module-level singleton: model load mehenga hai, process me ek hi baar.
_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformerEmbedder()
    return _EMBEDDER