from rank_bm25 import BM25Okapi

from app.models.chunk import Chunk
from app.models.retrieval_result import RetrievalResult


class BM25Store:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

        tokenized_corpus = [self._tokenize(chunk.content) for chunk in self.chunks]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if not self.chunks or self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]

        return [
            RetrievalResult(
                chunk=self.chunks[index],
                retrieval_score=float(scores[index]),
            )
            for index in ranked_indices
            if scores[index] > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
