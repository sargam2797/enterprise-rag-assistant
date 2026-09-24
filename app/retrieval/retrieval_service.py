import logging
import time

from app.embeddings.embedding_service import EmbeddingService
from app.lexical_search.bm25_store import BM25Store
from app.models.retrieval_result import RetrievalResult
from app.reranking.reranker import Reranker
from app.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion
from app.retrieval.relevance_filter import filter_by_reranker_score
from app.vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


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
        start_time = time.perf_counter()
        query_embedding = self.embedding_service.embed_text(question)

        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
        )

        bm25_results = self.bm25_store.search(
            query=question,
            limit=limit,
        )

        fused_results = reciprocal_rank_fusion([vector_results, bm25_results])

        reranked_results = self.reranker.rerank(
            question=question,
            results=fused_results,
        )

        final_results = filter_by_reranker_score(
            results=reranked_results,
            min_score=self.min_reranker_score,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if not final_results:
            logger.warning(
                "retrieval_no_relevant_context duration_ms=%.2f",
                duration_ms,
            )

        logger.info(
            "retrieval_completed "
            "vector_results=%d bm25_results=%d "
            "fused_results=%d final_results=%d "
            "duration_ms=%.2f",
            len(vector_results),
            len(bm25_results),
            len(fused_results),
            len(final_results),
            duration_ms,
        )

        return final_results
