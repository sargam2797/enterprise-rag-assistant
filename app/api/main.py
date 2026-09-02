from fastapi import FastAPI, Depends

from app.core.dependencies import get_rag_service

from pydantic import BaseModel

from app.services.rag_service import RAGService

app = FastAPI(
    title="Enterprise RAG Assistant",
    version="1.0.0",
)


class IngestRequest(BaseModel):
    file_path: str


class AskRequest(BaseModel):
    question: str


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
