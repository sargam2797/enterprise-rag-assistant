from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends

from app.api.schemas.ingestion import IngestRequest
from app.api.schemas.query import AskRequest
from app.core.dependencies import create_rag_service, get_rag_service

from pydantic import BaseModel

from app.core.logging_config import configure_logging
from app.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.rag_service = create_rag_service()
    yield


app = FastAPI(
    title="Enterprise RAG Assistant",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ingest")
def ingest_document(
    request: IngestRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    chunk_count = rag_service.ingest(request.file_path)

    return {
        "status": "success",
        "chunks_ingested": chunk_count,
    }


@app.post("/ask")
def ask_question(
    request: AskRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    answer = rag_service.answer(request.question)

    return {
        "answer": answer,
    }
