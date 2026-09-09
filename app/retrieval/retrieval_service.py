from app.embeddings.embedding_service import EmbeddingService
from app.lexical_search.bm25_store import BM25Store
from app.models.retrieval_result import RetrievalResult
from app.reranking.reranker import Reranker
from app.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion
from app.retrieval.relevance_filter import filter_by_reranker_score
from app.vector_store.qdrant_store import QdrantVectorStore


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        bm25_store: BM25Store,
        reranker: Reranker,
        min_reranker_score: float,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = reranker
        self.min_reranker_score = min_reranker_score

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

        bm25_results = self.bm25_store.search(
            query=question,
            limit=limit,
        )

        fused_results = reciprocal_rank_fusion([results, bm25_results])

        reranked_results = self.reranker.rerank(
            question=question,
            results=fused_results,
        )

        return filter_by_reranker_score(
            results=reranked_results,
            min_score=self.min_reranker_score,
        )
