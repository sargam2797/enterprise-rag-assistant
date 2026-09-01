# vector storage- Given this new query vector, find the vectors that are most semantically similar to it.


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from uuid import uuid4

from app.embeddings.embedding_service import EmbeddingService
from app.models.chunk import Chunk
from app.models.retrieval_result import RetrievalResult

embedding_service = EmbeddingService()


class QdrantVectorStore:
    def __init__(
        self,
        path: str = "qdrant_data",
        collection_name: str = "enterprise_documents",
        vector_size: int | None = embedding_service.dimension,
    ):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.vector_size = vector_size

        self._ensure_collection()

    # Whenever this vector store is created, make sure the required Qdrant collection exists.
    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    # we're converting our chunk+embedding into Qdrant point.
    # It will give us the distance and payload and becomes the context for the LLM
    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        "content": chunk.content,
                        "source": chunk.source,
                        "file_type": chunk.file_type,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_embedding: list[float] | None,
        limit: int = 3,
    ) -> list[RetrievalResult]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
        ).points

        return [
            RetrievalResult(
                chunk=Chunk(
                    content=result.payload["content"],
                    source=result.payload["source"],
                    file_type=result.payload["file_type"],
                    chunk_index=result.payload["chunk_index"],
                ),
                retrieval_score=result.score,
            )
            for result in results
        ]
