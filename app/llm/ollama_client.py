import ollama

from app.exceptions.exceptions import LLMUnavailableError


class OllamaClient:
    def __init__(self, model: str = "qwen3:4b"):
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = ollama.chat(
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
