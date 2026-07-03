from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from app.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    Embedding,
    EmbeddingValue,
    HealthResponse
)
from app.embedding import EmbeddingService


# Global embedding service instance
embedding_service = None


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


# Create FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="Local embedding service with dynamic model loading and Gemini-like API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if embedding_service is None:
        return HealthResponse(status="unhealthy", model="unknown", ready=False)

    loaded_models = embedding_service.get_loaded_models()
    is_healthy = len(loaded_models) > 0

    return HealthResponse(
        status="healthy" if is_healthy else "initializing",
        model=loaded_models[0] if loaded_models else "none",
        ready=is_healthy
    )


@app.get("/models")
async def list_models():
    """List available and loaded models"""
    if embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not ready"
        )

    info = embedding_service.get_model_info()
    return {
        "available_models": info["available_models"],
        "loaded_models": info["loaded_models"],
        "model_info": info["model_info"]
    }


@app.post("/v1/models/{model_name}:embedText", response_model=EmbeddingResponse)
async def embed_text(model_name: str, request: EmbeddingRequest):
    """
    Embed a single text string

    This endpoint matches Gemini's embedding API structure
    The model can be specified in the URL path or in the request body
    """
    if embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not ready"
        )

    try:
        # Use model from request if provided, otherwise use from URL
        api_model = request.model or f"models/{model_name}"

        # Resolve to actual model name
        actual_model = embedding_service.resolve_api_model(api_model)

        # Extract text from request
        text = " ".join(request.content.parts)

        # Generate embedding
        embeddings = embedding_service.encode([text], model_name=actual_model)

        # Format response in Gemini-like structure
        embedding_value = EmbeddingValue(values=embeddings[0])
        embedding = Embedding(value=embedding_value)

        return EmbeddingResponse(embedding={"embeddings": [embedding]})

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embedding: {str(e)}"
        )


@app.post("/v1/models/{model_name}:batchEmbedText", response_model=BatchEmbeddingResponse)
async def batch_embed_text(model_name: str, request: BatchEmbeddingRequest):
    """
    Embed multiple text strings

    This endpoint matches Gemini's batch embedding API structure
    The model can be specified in the URL path or in the request body
    """
    if embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not ready"
        )

    try:
        # Use model from request if provided, otherwise use from URL
        api_model = request.model or f"models/{model_name}"

        # Resolve to actual model name
        actual_model = embedding_service.resolve_api_model(api_model)

        # Extract texts from request
        texts = [" ".join(content.parts) for content in request.contents]

        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content provided for embedding"
            )

        # Generate embeddings
        embeddings = embedding_service.encode(texts, model_name=actual_model)

        # Format response in Gemini-like structure
        embedding_list = []
        for embedding in embeddings:
            embedding_value = EmbeddingValue(values=embedding)
            embedding_list.append(Embedding(value=embedding_value))

        return BatchEmbeddingResponse(embedding={"embeddings": embedding_list})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings: {str(e)}"
        )


@app.post("/v1/models/embedding-001:embedText", response_model=EmbeddingResponse)
async def embed_text_default(request: EmbeddingRequest):
    """Embed a single text string (default endpoint for compatibility)"""
    return await embed_text("embedding-001", request)


@app.post("/v1/models/embedding-001:batchEmbedText", response_model=BatchEmbeddingResponse)
async def batch_embed_text_default(request: BatchEmbeddingRequest):
    """Embed multiple text strings (default endpoint for compatibility)"""
    return await batch_embed_text("embedding-001", request)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    if embedding_service is None:
        return {
            "service": "Embedding Service",
            "status": "initializing",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "models": "/models",
                "docs": "/docs"
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
            "embed_text": "POST /v1/models/{model_name}:embedText",
            "batch_embed_text": "POST /v1/models/{model_name}:batchEmbedText",
            "docs": "/docs"
        },
        "available_models": model_info.get("available_models", []),
        "loaded_models": model_info.get("loaded_models", []),
        "model_info": model_info.get("model_info", {})
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
