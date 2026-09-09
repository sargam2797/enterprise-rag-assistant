from app.lexical_search.bm25_store import BM25Store
from app.models.chunk import Chunk


def create_chunk(content: str, index: int) -> Chunk:
    return Chunk(
        content=content,
        source="test.txt",
        file_type=".txt",
        chunk_index=index,
    )


def test_returns_empty_list_when_store_is_empty():
    store = BM25Store()

    results = store.search("Kafka consumer")

    assert results == []


def test_ranks_matching_chunk_first():
    store = BM25Store()

    kafka_chunk = create_chunk(
        "Kafka consumers read messages from topics",
        0,
    )

    chunks = [
        kafka_chunk,
        create_chunk(
            "PostgreSQL stores relational data in tables",
            1,
        ),
        create_chunk(
            "Redis provides in-memory caching",
            2,
        ),
        create_chunk(
            "Docker packages applications into containers",
            3,
        ),
        create_chunk(
            "Kubernetes orchestrates container workloads",
            4,
        ),
    ]

    store.add_chunks(chunks)

    results = store.search("Kafka consumers")

    assert len(results) > 0
    assert results[0].chunk == kafka_chunk
    assert results[0].retrieval_score > 0


def test_respects_search_limit():
    store = BM25Store()

    chunks = [
        create_chunk(
            "Kafka consumers process event messages",
            0,
        ),
        create_chunk(
            "Kafka producers publish records to topics",
            1,
        ),
        create_chunk(
            "PostgreSQL stores relational data",
            2,
        ),
        create_chunk(
            "Redis provides distributed caching",
            3,
        ),
        create_chunk(
            "Docker packages applications into containers",
            4,
        ),
    ]

    store.add_chunks(chunks)

    results = store.search(
        query="Kafka consumers",
        limit=1,
    )

    assert len(results) == 1
