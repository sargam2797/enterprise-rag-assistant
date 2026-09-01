from dataclasses import dataclass

from app.models.chunk import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    retrieval_score: float
    reranker_score: float | None = None
