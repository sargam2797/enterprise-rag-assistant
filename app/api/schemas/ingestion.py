from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    file_path: str = Field(min_length=1)
