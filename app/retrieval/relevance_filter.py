from app.models.retrieval_result import RetrievalResult


def filter_by_relevance(
    results: list[RetrievalResult],
    min_score: float,
) -> list[RetrievalResult]:
    return [result for result in results if result.retrieval_score >= min_score]
