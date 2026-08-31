from app.embeddings.embedding_service import EmbeddingService
from app.models.chunk import Chunk

service = EmbeddingService()


def test_embeds_single_text():
    embedding = service.embed_text("Kafka is an event streaming platform.")

    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)


def test_embeds_multiple_chunks():

    chunks = [
        Chunk(
            content="Kafka is an event streaming platform.",
            source="kafka.txt",
            file_type=".txt",
            chunk_index=0,
        ),
        Chunk(
            content="Consumers read messages from Kafka topics.",
            source="kafka.txt",
            file_type=".txt",
            chunk_index=1,
        ),
    ]

    embeddings = service.embed_chunks(chunks)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
