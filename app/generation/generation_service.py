import logging
import time

from app.llm.ollama_client import OllamaClient
from app.retrieval.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_client: OllamaClient,
    ):
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client

    def generate_answer(self, question: str) -> str:
        results = self.retrieval_service.retrieve(question)

        if not results:
            logger.info("generation_skipped reason=no_relevant_context")

            return "I don't have enough information to answer that question."

        context = "\n\n".join(result.chunk.content for result in results)

        prompt = f"""
Answer the question using only the provided context.

If the context does not contain enough information to answer the question,
say that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""
        start_time = time.perf_counter()

        answer = self.llm_client.generate(prompt)

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "generation_completed context_chunks=%d duration_ms=%.2f",
            len(results),
            duration_ms,
        )

        return answer
