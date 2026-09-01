from dataclasses import dataclass

from app.models.chunk import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
