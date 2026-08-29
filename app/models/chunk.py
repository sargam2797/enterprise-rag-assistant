from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    source: str
    file_type: str
    chunk_index: int
