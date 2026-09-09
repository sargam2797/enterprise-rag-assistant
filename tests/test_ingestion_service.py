from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.ingestion_service import IngestionService
from app.lexical_search.bm25_store import BM25Store
from app.lexical_search.bm25_store import BM25Store
from app.vector_store.qdrant_store import QdrantVectorStore


def test_ingests_document_into_qdrant(tmp_path):
    document_path = tmp_path / "kafka.txt"

    document_path.write_text(
        "Kafka is an event streaming platform.\n\n"
        "Kafka consumers read messages from topics.",
        encoding="utf-8",
    )

    embedding_service = EmbeddingService()
    bm25_store = BM25Store()

    vector_store = QdrantVectorStore(
        vector_size=embedding_service.dimension,
        path=str(tmp_path / "qdrant"),
        collection_name="test_documents",
    )

    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    chunk_count = ingestion_service.ingest(
        file_path=str(document_path),
        chunk_size=50,
        chunk_overlap=10,
    )

    collection_info = vector_store.client.get_collection("test_documents")

    assert chunk_count > 0
    assert collection_info.points_count == chunk_count
