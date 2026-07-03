from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from app.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    ModelsListResponse
)
from app.embedding import EmbeddingService


# Global embedding service instance
embedding_service = None


# OpenAPI documentation for LLM understanding
API_DESCRIPTION = """
## Embedding Service API

A local text embedding service using sentence-transformers models.

### Features
- Dynamic model loading (models loaded on-demand)
- Local model caching (stored in `models/` directory)
- Single endpoint for both single and batch embeddings
- CPU-only operation (no GPU required)

### Usage
1. Send POST request to `/v1/embed` with your texts
2. Specify model name (optional, defaults to `all-MiniLM-L6-v2`)
3. Receive embeddings as list of float vectors

### Example Request
\`\`\`json
{
  "model": "models/embedding-001",
  "texts": ["Hello, world!", "Another text"]
}
\`\`\`

### Example Response
\`\`\`json
{
  "embeddings": [
    [0.1, -0.2, 0.3, ...],
    [0.4, 0.5, -0.6, ...]
  ]
}
\`\`\`

### Available Models
- `models/embedding-001` → all-MiniLM-L6-v2 (384 dimensions)
- `models/all-MiniLM-L6-v2` → all-MiniLM-L6-v2 (384 dimensions)

More models can be added by updating the MODEL_REGISTRY in `app/embedding.py`.

### Model Information
- **Default Model**: all-MiniLM-L6-v2
- **Embedding Dimension**: 384
- **Max Sequence Length**: 256 tokens
- **Normalization**: L2 normalized by default
- **Storage**: Models cached in `models/` directory after first download
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global embedding_service

    # Startup
    print("Starting Embedding Service...")
    embedding_service = EmbeddingService(cache_dir="models")
    print("Embedding Service is ready!")
    print("Models will be loaded on-demand based on API requests")

    yield

    # Shutdown
    print("Shutting down Embedding Service...")


# Create FastAPI app with enhanced OpenAPI documentation
app = FastAPI(
    title="Embedding Service",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    tags=[
        {
            "name": "embeddings",
            "description": "Generate text embeddings using various models"
        },
        {
            "name": "health",
            "description": "Service health and status checks"
        },
        {
            "name": "models",
            "description": "List available and loaded models"
        }
    ],
    contact={
        "name": "Embedding Service",
        "url": "http://localhost:8800",
    },
    license_info={
        "name": "MIT",
    },
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is running and if any models are loaded.",
    tags=["health"],
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "model": "all-MiniLM-L6-v2",
                        "ready": True
                    }
                }
            }
        }
    }
)
async def health_check():
    """Health check endpoint

    Returns the current status of the embedding service and whether models are loaded.
    """
    if embedding_service is None:
        return HealthResponse(status="unhealthy", model="unknown", ready=False)

    loaded_models = embedding_service.get_loaded_models()
    is_healthy = len(loaded_models) > 0

    return HealthResponse(
        status="healthy" if is_healthy else "initializing",
        model=loaded_models[0] if loaded_models else "none",
        ready=is_healthy
    )


@app.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List available models",
    description="Get a list of all available models and currently loaded models with their metadata.",
    tags=["models"],
    responses={
        200: {
            "description": "List of models",
            "content": {
                "application/json": {
                    "example": {
                        "available_models": ["models/embedding-001", "models/all-MiniLM-L6-v2"],
                        "loaded_models": ["all-MiniLM-L6-v2"],
                        "model_info": {
                            "all-MiniLM-L6-v2": {
                                "model_name": "all-MiniLM-L6-v2",
                                "max_seq_length": 256,
                                "embedding_dimension": 384
                            }
                        }
                    }
                }
            }
        }
    }
)
async def list_models():
    """List available and loaded models

    Returns information about registered models and which are currently loaded in memory.
    Includes model metadata like embedding dimension and max sequence length.
    """
    if embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not ready"
        )

    info = embedding_service.get_model_info()
    return ModelsListResponse(
        available_models=info["available_models"],
        loaded_models=info["loaded_models"],
        model_info=info["model_info"]
    )


@app.post(
    "/v1/embed",
    response_model=EmbeddingResponse,
    summary="Generate embeddings",
    description="""
Generate embeddings for one or more text strings.

This endpoint handles both single and batch requests seamlessly.
Simply pass an array of texts and receive corresponding embeddings.

**Single text example:**
```json
{"model": "models/embedding-001", "texts": ["hello"]}
```

**Batch example:**
```json
{"model": "models/embedding-001", "texts": ["hello", "world"]}
```

The model will be downloaded and cached on first use if not already present.
    """,
    tags=["embeddings"],
    responses={
        200: {
            "description": "Embeddings generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "embeddings": [
                            [0.1, -0.2, 0.3, -0.4, 0.5]
                        ]
                    }
                }
            }
        },
        400: {
            "description": "Invalid request (empty texts array)"
        },
        500: {
            "description": "Internal server error (model loading failed)"
        }
    }
)
async def embed(request: EmbeddingRequest):
    """
    Generate embeddings for text strings

    Handles both single and batch requests:
    - Single text: `{"texts": ["hello"]}` → `{"embeddings": [[...]]}`
    - Batch text: `{"texts": ["hello", "world"]}` → `{"embeddings": [[...], [...]]}`
    """
    if embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not ready"
        )

    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="texts field cannot be empty"
        )

    try:
        # Resolve model name
        actual_model = embedding_service.resolve_api_model(request.model)

        # Generate embeddings
        embeddings = embedding_service.encode(request.texts, model_name=actual_model)

        return EmbeddingResponse(embeddings=embeddings)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings: {str(e)}"
        )


@app.get(
    "/",
    summary="Service information",
    description="Get service status, available endpoints, and model information",
    tags=["health"]
)
async def root():
    """Root endpoint with service information

    Returns service status, available endpoints, and model information.
    Useful for checking if the service is running and what models are available.
    """
    if embedding_service is None:
        return {
            "service": "Embedding Service",
            "status": "initializing",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "models": "/models",
                "embed": "POST /v1/embed",
                "docs": "/docs",
                "openapi": "/openapi.json"
            }
        }

    model_info = embedding_service.get_model_info()
    return {
        "service": "Embedding Service",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "embed": "POST /v1/embed",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "available_models": model_info.get("available_models", []),
        "loaded_models": model_info.get("loaded_models", []),
        "model_info": model_info.get("model_info", {})
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
