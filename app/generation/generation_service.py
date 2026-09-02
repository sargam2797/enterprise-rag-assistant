from app.llm.ollama_client import OllamaClient
from app.retrieval.retrieval_service import RetrievalService


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

        return self.llm_client.generate(prompt)
