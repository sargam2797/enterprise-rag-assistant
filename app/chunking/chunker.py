from app.models.chunk import Chunk
from app.models.document import Document


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    chunk_index = 0

    # Process the document until we reach the end of its content
    while start < len(document.content):
        end = start + chunk_size

        # Python slicing: start is included, end is excluded
        chunk_content = document.content[start:end]

        # Preserve source metadata so retrieved chunks remain traceable
        chunks.append(
            Chunk(
                content=chunk_content,
                source=document.source,
                file_type=document.file_type,
                chunk_index=chunk_index,
            )
        )

        # Avoid creating an unnecessary final overlap-only chunk
        if end >= len(document.content):
            break

        # Move forward while preserving some previous context
        start = end - chunk_overlap
        chunk_index += 1

    return chunks
