from app.models.chunk import Chunk
from app.models.retrieval_result import RetrievalResult
from app.retrieval.relevance_filter import filter_by_relevance


def test_filters_results_below_minimum_score():
    chunk1 = Chunk(
        content="Kafka consumers read messages from topics.",
        source="kafka.txt",
        file_type=".txt",
        chunk_index=0,
    )

    chunk2 = Chunk(
        content="PostgreSQL is a relational database.",
        source="postgres.txt",
        file_type=".txt",
        chunk_index=0,
    )

    results = [
        RetrievalResult(chunk=chunk1, retrieval_score=0.82),
        RetrievalResult(chunk=chunk2, retrieval_score=0.31),
    ]

    filtered_results = filter_by_relevance(
        results=results,
        min_score=0.60,
    )

    assert len(filtered_results) == 1
    assert filtered_results[0].chunk.source == "kafka.txt"


def test_returns_empty_list_when_nothing_is_relevant():
    chunk = Chunk(
        content="Docker packages applications into containers.",
        source="docker.txt",
        file_type=".txt",
        chunk_index=0,
    )

    results = [
        RetrievalResult(chunk=chunk, retrieval_score=0.22),
    ]

    filtered_results = filter_by_relevance(
        results=results,
        min_score=0.60,
    )

    assert filtered_results == []
