from app.models.retrieval_result import RetrievalResult
from app.retrieval.retrieval_service import RetrievalService
from evaluation.dataset import EvaluationCase


def is_hit(
    results: list[RetrievalResult],
    expected_source: str,
    k: int,
) -> bool:
    top_k_results = results[:k]

    return any(result.chunk.source == expected_source for result in top_k_results)


def evaluate_retrieval(
    retrieval_service: RetrievalService,
    cases: list[EvaluationCase],
    k: int = 3,
) -> dict:
    relevant_cases = [
        case
        for case in cases
        if case.should_answer and case.expected_source is not None
    ]

    hits = 0

    for case in relevant_cases:
        results = retrieval_service.retrieve(
            question=case.question,
            limit=k,
        )

        if is_hit(
            results=results,
            expected_source=case.expected_source,
            k=k,
        ):
            hits += 1

    total = len(relevant_cases)

    hit_rate = hits / total if total > 0 else 0.0

    return {
        "total": total,
        "hits": hits,
        "hit_rate": hit_rate,
        "k": k,
    }


def evaluate_rejection(
    retrieval_service: RetrievalService,
    cases: list[EvaluationCase],
) -> dict:
    irrelevant_cases = [case for case in cases if not case.should_answer]

    correctly_rejected = 0

    for case in irrelevant_cases:
        results = retrieval_service.retrieve(
            question=case.question,
        )

        if not results:
            correctly_rejected += 1

    total = len(irrelevant_cases)

    rejection_rate = correctly_rejected / total if total > 0 else 0.0

    return {
        "total": total,
        "correctly_rejected": correctly_rejected,
        "rejection_rate": rejection_rate,
    }
