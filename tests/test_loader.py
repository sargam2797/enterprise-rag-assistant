from pathlib import Path

import pytest

from app.exceptions.exceptions import (
    DocumentNotFoundError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.loader import load_document


def test_load_text_document(tmp_path: Path):
    file = tmp_path / "sample.txt"
    file.write_text("Kafka is an event streaming platform.")

    document = load_document(str(file))

    assert document.content == "Kafka is an event streaming platform."
    assert document.source == str(file)
    assert document.file_type == ".txt"


def test_raises_error_when_file_does_not_exist():
    with pytest.raises(DocumentNotFoundError):
        load_document("missing.txt")


def test_raises_error_for_unsupported_file_type(tmp_path: Path):
    file = tmp_path / "sample.pdf"
    file.write_text("Some content")

    with pytest.raises(UnsupportedDocumentTypeError):
        load_document(str(file))
