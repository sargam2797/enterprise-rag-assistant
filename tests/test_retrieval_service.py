from app.embeddings.embedding_service import EmbeddingService
from app.lexical_search.bm25_store import BM25Store
from app.models.chunk import Chunk
from app.reranking.reranker import Reranker
from app.retrieval.retrieval_service import RetrievalService
from app.vector_store.qdrant_store import QdrantVectorStore


def test_retrieves_relevant_chunks(tmp_path):
    global result
    embedding_service = EmbeddingService()
    reranker = Reranker()
    bm25_store = BM25Store()

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
    bm25_store.add_chunks(chunks)

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_store=bm25_store,
        reranker=reranker,
        min_reranker_score=0.00,
    )

    results = retrieval_service.retrieve(
        "How do Kafka consumers receive messages?",
        limit=2,
    )

    assert len(results) >= 1
    assert results[0].chunk.source == "kafka.txt"
