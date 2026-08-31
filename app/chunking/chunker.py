from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    split_texts = text_splitter.split_text(document.content)

    return [
        Chunk(
            content=text,
            source=document.source,
            file_type=document.file_type,
            chunk_index=index,
        )
        for index, text in enumerate(split_texts)
    ]
