from sentence_transformers import CrossEncoder

from app.models.retrieval_result import RetrievalResult


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [[question, result.chunk.content] for result in results]

        scores = self.model.predict(pairs)

        ranked_results = sorted(
            zip(results, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [result for result, _ in ranked_results]
