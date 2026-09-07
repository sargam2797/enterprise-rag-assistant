from fastapi import Request

from app.embeddings.embedding_service import EmbeddingService
from app.generation.generation_service import GenerationService
from app.ingestion.ingestion_service import IngestionService
from app.llm.ollama_client import OllamaClient
from app.reranking.reranker import Reranker
from app.retrieval.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
from app.vector_store.qdrant_store import QdrantVectorStore
from app.core.config import settings


def create_rag_service() -> RAGService:
    embedding_service = EmbeddingService(model_name=settings.embedding_model)

    vector_store = QdrantVectorStore(
        vector_size=embedding_service.dimension,
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
    )

    reranker = Reranker(model_name=settings.reranker_model)

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        reranker=reranker,
        min_score=settings.retrieval_min_score,
    )

    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    llm_client = OllamaClient(model=settings.llm_model)

    generation_service = GenerationService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
    )

    return RAGService(
        ingestion_service=ingestion_service,
        generation_service=generation_service,
    )


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service
