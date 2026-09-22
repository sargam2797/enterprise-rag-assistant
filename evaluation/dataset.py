from dataclasses import dataclass


@dataclass
class EvaluationCase:
    question: str
    expected_source: str | None
    should_answer: bool


EVALUATION_CASES = [
    EvaluationCase(
        question="How do Kafka consumers receive messages?",
        expected_source="data/kafka.txt",
        should_answer=True,
    ),
    EvaluationCase(
        question="How do consumer groups distribute work?",
        expected_source="data/kafka.txt",
        should_answer=True,
    ),
    EvaluationCase(
        question="What is the capital of Japan?",
        expected_source=None,
        should_answer=False,
    ),
]
