from app.embeddings.embedding_service import EmbeddingService
from app.models.retrieval_result import RetrievalResult
from app.reranking.reranker import Reranker
from app.retrieval.relevance_filter import filter_by_relevance
from app.vector_store.qdrant_store import QdrantVectorStore


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        reranker: Reranker,
        min_score: float = 0.60,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker
        self.min_score = min_score

    def retrieve(
        self,
        question: str,
        limit: int = 3,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedding_service.embed_text(question)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
        )

        relevant_results = filter_by_relevance(
            results=results,
            min_score=self.min_score,
        )

        return self.reranker.rerank(
            question=question,
            results=relevant_results,
        )
