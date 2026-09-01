from app.embeddings.embedding_service import EmbeddingService
from app.models.chunk import Chunk
from app.vector_store.qdrant_store import QdrantVectorStore

embedding_service = EmbeddingService()


def test_stores_chunks_in_qdrant(tmp_path):
    chunks = [
        Chunk(
            content="Kafka is an event streaming platform.",
            source="kafka.txt",
            file_type=".txt",
            chunk_index=0,
        ),
        Chunk(
            content="Consumers read messages from Kafka topics.",
            source="kafka.txt",
            file_type=".txt",
            chunk_index=1,
        ),
    ]

    embeddings = embedding_service.embed_chunks(chunks)

    store = QdrantVectorStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_documents",
        vector_size=embedding_service.dimension,
    )

    store.add_chunks(chunks, embeddings)

    collection_info = store.client.get_collection("test_documents")

    assert collection_info.points_count == 2


def test_multiple_ingestions_do_not_overwrite_existing_chunks(tmp_path):
    store = QdrantVectorStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_documents",
        vector_size=embedding_service.dimension,
    )

    first_chunks = [
        Chunk(
            content="Kafka is an event streaming platform.",
            source="kafka.txt",
            file_type=".txt",
            chunk_index=0,
        )
    ]

    second_chunks = [
        Chunk(
            content="Java is a programming language.",
            source="java.txt",
            file_type=".txt",
            chunk_index=0,
        )
    ]

    first_embeddings = embedding_service.embed_chunks(first_chunks)
    second_embeddings = embedding_service.embed_chunks(second_chunks)

    store.add_chunks(first_chunks, first_embeddings)
    store.add_chunks(second_chunks, second_embeddings)

    collection_info = store.client.get_collection("test_documents")

    assert collection_info.points_count == 2


def test_retrieves_semantically_similar_chunk(tmp_path):
    store = QdrantVectorStore(
        vector_size=embedding_service.dimension,
        path=str(tmp_path / "qdrant"),
        collection_name="test_documents",
    )

    chunks = [
        Chunk(
            content="Kafka consumers read messages from topics.",
            source="kafka.txt",
            file_type=".txt",
            chunk_index=0,
        ),
        Chunk(
            content="PostgreSQL is a relational database.",
            source="postgres.txt",
            file_type=".txt",
            chunk_index=0,
        ),
        Chunk(
            content="Docker containers package applications and their dependencies.",
            source="docker.txt",
            file_type=".txt",
            chunk_index=0,
        ),
    ]

    embeddings = embedding_service.embed_chunks(chunks)

    store.add_chunks(chunks, embeddings)

    query_embedding = embedding_service.embed_text(
        "How do Kafka consumers receive messages?"
    )

    results = store.search(
        query_embedding=query_embedding,
        limit=1,
    )

    assert len(results) == 1
    assert results[0].chunk.content == "Kafka consumers read messages from topics."
    assert results[0].score == 0.8396311788805287
