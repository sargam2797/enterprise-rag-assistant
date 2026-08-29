from pathlib import Path

from app.models.document import Document

SUPPORTED_EXTENSIONS = {".txt", ".md"}


# Loads a file and converts it into our internal Document model
def load_document(file_path: str) -> Document:

    # Convert the string file path into a Path object for easier file operations
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    content = path.read_text(encoding="utf-8")

    return Document(content=content, source=str(path), file_type=str(path.suffix))
