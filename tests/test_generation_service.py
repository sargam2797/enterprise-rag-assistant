from unittest.mock import Mock

from app.generation.generation_service import GenerationService
from app.models.chunk import Chunk
from app.models.retrieval_result import RetrievalResult


def test_generates_answer_using_retrieved_context():
    # Arrange
    mock_retrieval_service = Mock()
    mock_llm_client = Mock()

    mock_retrieval_service.retrieve.return_value = [
        RetrievalResult(
            chunk=Chunk(
                content="Kafka consumers read messages from topics.",
                source="kafka.txt",
                file_type=".txt",
                chunk_index=0,
            ),
            retrieval_score=0.82,
            reranker_score=5.4,
        )
    ]

    mock_llm_client.generate.return_value = "Kafka consumers read messages from topics."

    service = GenerationService(
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    # Act
    answer = service.generate_answer("How do Kafka consumers receive messages?")

    # Assert
    assert answer == "Kafka consumers read messages from topics."

    mock_retrieval_service.retrieve.assert_called_once_with(
        "How do Kafka consumers receive messages?"
    )

    mock_llm_client.generate.assert_called_once()


def test_does_not_call_llm_when_no_relevant_context():
    mock_retrieval_service = Mock()
    mock_llm_client = Mock()

    mock_retrieval_service.retrieve.return_value = []

    service = GenerationService(
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    answer = service.generate_answer("What is the best pizza in Pune?")

    assert answer == ("I don't have enough information to answer that question.")

    mock_llm_client.generate.assert_not_called()


def test_includes_retrieved_context_in_llm_prompt():
    mock_retrieval_service = Mock()
    mock_llm_client = Mock()

    mock_retrieval_service.retrieve.return_value = [
        RetrievalResult(
            chunk=Chunk(
                content="Consumer groups distribute Kafka partitions.",
                source="kafka.txt",
                file_type=".txt",
                chunk_index=0,
            ),
            retrieval_score=0.80,
            reranker_score=4.8,
        )
    ]

    mock_llm_client.generate.return_value = "Generated answer"

    service = GenerationService(
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    service.generate_answer("How do consumer groups work?")

    prompt = mock_llm_client.generate.call_args.args[0]

    assert "Consumer groups distribute Kafka partitions." in prompt
    assert "How do consumer groups work?" in prompt
