"""Protocol defining the embedding contract. All embedder implementations must satisfy this."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Contract for text embedding backends.

    Implementations must be swappable without changing any call site.
    See: src/rag/ingestion/embedder.py for OpenAIEmbedder and BGEEmbedder.
    """

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a normalised float vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one vector per input, in order."""
        ...
