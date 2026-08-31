from sentence_transformers import SentenceTransformer

from app.models.chunk import Chunk


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        # Load the embedding model once when the service is created
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        # Convert a single piece of text into a semantic vector
        embedding = self.model.encode(text)

        # SentenceTransformer returns a NumPy array.
        # Convert it to a normal Python list for easier use by other components.
        return embedding.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        # Extract only the text because the embedding model works on text,
        # not our Chunk objects.
        texts = [chunk.content for chunk in chunks]

        embeddings = self.model.encode(texts)

        return embeddings.tolist()
