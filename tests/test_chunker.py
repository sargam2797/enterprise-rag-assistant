import pytest

from app.chunking.chunker import chunk_document
from app.exceptions.exceptions import InvalidChunkConfigurationError
from app.models.document import Document


def test_chunks_document_with_overlap():
    document = Document(
        content="ABCDEFGHIJKLMNO",
        source="sample.txt",
        file_type=".txt",
    )

    chunks = chunk_document(
        document=document,
        chunk_size=10,
        chunk_overlap=3,
    )

    assert len(chunks) == 2

    assert chunks[0].content == "ABCDEFGHIJ"
    assert chunks[0].chunk_index == 0

    assert chunks[1].content == "HIJKLMNO"
    assert chunks[1].chunk_index == 1


def test_preserves_document_metadata():
    document = Document(
        content="Hello world",
        source="docs/sample.md",
        file_type=".md",
    )

    chunks = chunk_document(
        document=document,
        chunk_size=5,
        chunk_overlap=1,
    )

    assert chunks[0].source == "docs/sample.md"
    assert chunks[0].file_type == ".md"


def test_raises_error_when_chunk_size_is_zero():
    document = Document(
        content="Hello",
        source="sample.txt",
        file_type=".txt",
    )

    with pytest.raises(InvalidChunkConfigurationError):
        chunk_document(document, chunk_size=0)


def test_raises_error_when_overlap_is_greater_than_chunk_size():
    document = Document(
        content="Hello",
        source="sample.txt",
        file_type=".txt",
    )

    with pytest.raises(InvalidChunkConfigurationError):
        chunk_document(
            document,
            chunk_size=5,
            chunk_overlap=5,
        )
