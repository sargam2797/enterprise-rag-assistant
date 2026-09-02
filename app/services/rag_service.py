from app.generation.generation_service import GenerationService
from app.ingestion.ingestion_service import IngestionService


class RAGService:
    def __init__(
        self,
        ingestion_service: IngestionService,
        generation_service: GenerationService,
    ):
        self.ingestion_service = ingestion_service
        self.generation_service = generation_service

    def ingest(self, file_path: str) -> int:
        return self.ingestion_service.ingest(file_path)

    def answer(self, question: str) -> str:
        return self.generation_service.generate_answer(question)
