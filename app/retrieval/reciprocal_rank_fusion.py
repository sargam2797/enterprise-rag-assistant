from app.models.retrieval_result import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    scores: dict[tuple[str, int], float] = {}
    results_by_key: dict[tuple[str, int], RetrievalResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = (
                result.chunk.source,
                result.chunk.chunk_index,
            )

            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            results_by_key[key] = result

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        RetrievalResult(
            chunk=results_by_key[key].chunk,
            retrieval_score=scores[key],
        )
        for key in ranked_keys
    ]
