from unittest.mock import Mock

from app.models.chunk import Chunk
from app.models.retrieval_result import RetrievalResult
from evaluation.dataset import EvaluationCase
from evaluation.retrieval_evaluator import is_hit, evaluate_retrieval


def create_result(source: str) -> RetrievalResult:
    chunk = Chunk(
        content="test content",
        source=source,
        file_type=".txt",
        chunk_index=0,
    )

    return RetrievalResult(
        chunk=chunk,
        retrieval_score=1.0,
        reranker_score=1.0,
    )


def test_returns_true_when_expected_source_is_in_top_k():
    results = [
        create_result("redis.txt"),
        create_result("data/kafka.txt"),
        create_result("postgres.txt"),
    ]

    assert is_hit(
        results=results,
        expected_source="data/kafka.txt",
        k=3,
    )


def test_returns_false_when_expected_source_is_not_present():
    results = [
        create_result("redis.txt"),
        create_result("postgres.txt"),
        create_result("docker.txt"),
    ]

    assert not is_hit(
        results=results,
        expected_source="data/kafka.txt",
        k=3,
    )


def test_returns_false_when_expected_source_is_below_k():
    results = [
        create_result("redis.txt"),
        create_result("postgres.txt"),
        create_result("docker.txt"),
        create_result("data/kafka.txt"),
    ]

    assert not is_hit(
        results=results,
        expected_source="data/kafka.txt",
        k=3,
    )


def test_calculates_hit_rate():
    retrieval_service = Mock()

    retrieval_service.retrieve.side_effect = [
        [create_result("data/kafka.txt")],
        [create_result("data/postgres.txt")],
    ]

    cases = [
        EvaluationCase(
            question="How do Kafka consumers work?",
            expected_source="data/kafka.txt",
            should_answer=True,
        ),
        EvaluationCase(
            question="What is Redis?",
            expected_source="data/redis.txt",
            should_answer=True,
        ),
        EvaluationCase(
            question="What is the capital of Japan?",
            expected_source=None,
            should_answer=False,
        ),
    ]

    result = evaluate_retrieval(
        retrieval_service=retrieval_service,
        cases=cases,
        k=3,
    )

    assert result["total"] == 2
    assert result["hits"] == 1
    assert result["hit_rate"] == 0.5
    assert result["k"] == 3
