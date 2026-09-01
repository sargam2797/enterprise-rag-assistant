from app.embeddings.embedding_service import EmbeddingService
from app.models.chunk import Chunk
from app.retrieval.retrieval_service import RetrievalService
from app.vector_store.qdrant_store import QdrantVectorStore


def test_retrieves_relevant_chunks(tmp_path):
    embedding_service = EmbeddingService()

    vector_store = QdrantVectorStore(
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
    ]

    embeddings = embedding_service.embed_chunks(chunks)
    vector_store.add_chunks(chunks, embeddings)

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        min_score=0.50,
    )

    results = retrieval_service.retrieve(
        "How do Kafka consumers receive messages?",
        limit=2,
    )

    assert len(results) >= 1
    assert results[0].chunk.source == "kafka.txt"
