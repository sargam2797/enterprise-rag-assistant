from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.main import app
from app.core.dependencies import get_rag_service

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ingests_document():
    mock_rag_service = Mock()
    mock_rag_service.ingest.return_value = 3

    def override_rag_service():
        return mock_rag_service

    app.dependency_overrides[get_rag_service] = override_rag_service

    response = client.post(
        "/ingest",
        json={
            "file_path": "data/kafka.txt",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "chunks_ingested": 3,
    }

    mock_rag_service.ingest.assert_called_once_with("data/kafka.txt")

    app.dependency_overrides.clear()


def test_answers_question():
    mock_rag_service = Mock()
    mock_rag_service.answer.return_value = "Kafka consumers read messages from topics."

    def override_rag_service():
        return mock_rag_service

    app.dependency_overrides[get_rag_service] = override_rag_service

    response = client.post(
        "/ask",
        json={"question": "How do Kafka consumers receive messages?"},
    )

    assert response.status_code == 200

    assert response.json() == {"answer": "Kafka consumers read messages from topics."}

    mock_rag_service.answer.assert_called_once_with(
        "How do Kafka consumers receive messages?"
    )

    app.dependency_overrides.clear()
