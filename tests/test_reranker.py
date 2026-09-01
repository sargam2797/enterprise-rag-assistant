from app.models.chunk import Chunk
from app.models.retrieval_result import RetrievalResult
from app.reranking.reranker import Reranker

reranker = Reranker()


def test_reranks_more_relevant_chunk_first():
    results = [
        RetrievalResult(
            chunk=Chunk(
                content="PostgreSQL is a relational database.",
                source="postgres.txt",
                file_type=".txt",
                chunk_index=0,
            ),
            retrieval_score=0.75,
        ),
        RetrievalResult(
            chunk=Chunk(
                content="Kafka consumers read messages from topics.",
                source="kafka.txt",
                file_type=".txt",
                chunk_index=0,
            ),
            retrieval_score=0.70,
        ),
    ]

    reranked_results = reranker.rerank(
        question="How do Kafka consumers receive messages?",
        results=results,
    )

    assert len(reranked_results) == 2

    # Kafka should move to the first position after reranking.
    assert reranked_results[0].chunk.source == "kafka.txt"

    # Original retrieval score should still be preserved.
    assert reranked_results[0].retrieval_score == 0.70
