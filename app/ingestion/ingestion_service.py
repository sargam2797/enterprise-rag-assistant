from app.chunking.chunker import chunk_document
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.loader import load_document
from app.vector_store.qdrant_store import QdrantVectorStore


class IngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> int:
        document = load_document(file_path)

        chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        embeddings = self.embedding_service.embed_chunks(chunks)

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return len(chunks)
