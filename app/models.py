from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model matching Gemini's embedding API structure"""

    class Content(BaseModel):
        parts: List[str]

    content: Content
    model: Optional[str] = Field(default="models/embedding-001", description="Model identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "models/embedding-001",
                "content": {
                    "parts": ["Your text string here"]
                }
            }
        }


class EmbeddingValue(BaseModel):
    """Individual embedding value"""

    values: List[float]


class Embedding(BaseModel):
    """Embedding result for a single content item"""

    value: EmbeddingValue


class EmbeddingResponse(BaseModel):
    """Response model matching Gemini's embedding API structure"""

    class EmbeddingResult(BaseModel):
        embeddings: List[Embedding]

    embedding: EmbeddingResult


class BatchEmbeddingRequest(BaseModel):
    """Batch embedding request for multiple texts"""

    class Content(BaseModel):
        parts: List[str]

    contents: List[Content]
    model: Optional[str] = Field(default="models/embedding-001", description="Model identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "models/embedding-001",
                "contents": [
                    {"parts": ["First text string"]},
                    {"parts": ["Second text string"]}
                ]
            }
        }


class BatchEmbeddingResponse(BaseModel):
    """Batch embedding response"""

    class EmbeddingResult(BaseModel):
        embeddings: List[Embedding]

    embedding: EmbeddingResult


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model: str
    ready: bool
