from app.embeddings.embedding_service import EmbeddingService
from app.generation.generation_service import GenerationService
from app.ingestion.ingestion_service import IngestionService
from app.llm.ollama_client import OllamaClient
from app.reranking.reranker import Reranker
from app.retrieval.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
from app.vector_store.qdrant_store import QdrantVectorStore

embedding_service = EmbeddingService()

vector_store = QdrantVectorStore(
    vector_size=embedding_service.dimension,
    path="qdrant_data",
    collection_name="enterprise_documents",
)

reranker = Reranker()

retrieval_service = RetrievalService(
    embedding_service=embedding_service,
    vector_store=vector_store,
    reranker=reranker,
    min_score=0.30,
)

ingestion_service = IngestionService(
    embedding_service=embedding_service,
    vector_store=vector_store,
)

llm_client = OllamaClient()

generation_service = GenerationService(
    retrieval_service=retrieval_service,
    llm_client=llm_client,
)

rag_service = RAGService(
    ingestion_service=ingestion_service,
    generation_service=generation_service,
)


def get_rag_service() -> RAGService:
    return rag_service
