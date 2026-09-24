from app.llm.ollama_client import OllamaClient


def test_generates_response():
    client = OllamaClient(model="qwen3:4b", base_url="http://localhost:11434")

    response = client.generate("Reply with exactly the word: hello")

    assert isinstance(response, str)
    assert len(response) > 0
