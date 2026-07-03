from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for text embeddings"""

    model: str = Field(default="models/embedding-001", description="Model identifier")
    texts: List[str] = Field(..., description="List of text strings to embed")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "models/embedding-001",
                "texts": ["Your text string here"]
            }
        }


class EmbeddingResponse(BaseModel):
    """Response model for embeddings"""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [
                    [0.1, -0.2, 0.3, -0.4, 0.5]
                ]
            }
        }


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    max_seq_length: int
    embedding_dimension: int


class ModelsListResponse(BaseModel):
    """Response for model listing endpoint"""
    available_models: List[str]
    loaded_models: List[str]
    model_info: dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    ready: bool
