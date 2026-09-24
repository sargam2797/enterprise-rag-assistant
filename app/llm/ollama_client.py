import ollama

from app.exceptions.exceptions import LLMUnavailableError


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = ollama.Client(host=base_url)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            return response["message"]["content"]

        except ConnectionError as error:
            raise LLMUnavailableError(
                "LLM service is currently unavailable."
            ) from error
