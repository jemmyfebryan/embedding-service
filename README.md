# Embedding Service

A local embedding service powered by FastAPI and `sentence-transformers` with dynamic model loading support.

## Features

- FastAPI-based REST API
- **Dynamic model loading** - Models loaded on-demand from API requests
- **Local model caching** - Models stored in `models/` directory, no re-downloading
- **Multi-model support** - Easy to add new embedding models
- **Simplified API** - Single endpoint for both single and batch embeddings
- **CPU-only PyTorch** - Lightweight, no CUDA dependencies
- **Rich OpenAPI docs** - Auto-generated for LLM consumption

## Quick Start

### 1. Install Dependencies

```bash
make install
```

### 2. Start the Service

```bash
make start
```

### 3. Check Status

```bash
make status
```

### 4. Test the Service

```bash
python test_service.py
```

Or visit the interactive docs:
```
http://localhost:8800/docs       # Swagger UI
http://localhost:8800/redoc      # ReDoc
http://localhost:8800/openapi.json  # OpenAPI spec for LLMs
```

## API Endpoints

### Health Check
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model": "all-MiniLM-L6-v2",
  "ready": true
}
```

### List Available Models
```
GET /models
```

Response:
```json
{
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
```

### Generate Embeddings
```
POST /v1/embed
```

**Single text:**
```json
{
  "model": "models/embedding-001",
  "texts": ["Your text here"]
}
```

**Batch texts:**
```json
{
  "model": "models/embedding-001",
  "texts": ["First text", "Second text", "Third text"]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.1, -0.2, 0.3, ...],
    [0.4, 0.5, -0.6, ...],
    [0.7, -0.8, 0.9, ...]
  ]
}
```

## OpenAPI Documentation

The service provides comprehensive OpenAPI documentation for LLM consumption:

```bash
# Get OpenAPI spec
curl http://localhost:8800/openapi.json

# Or visit interactive docs
http://localhost:8800/docs
http://localhost:8800/redoc
```

You can pass the OpenAPI spec directly to an LLM to help it understand:
- All available endpoints
- Request/response schemas
- Required vs optional fields
- Example usage
- Error responses

## Available Models

Edit `app/embedding.py` to add more models:

```python
MODEL_REGISTRY = {
    "models/embedding-001": "all-MiniLM-L6-v2",
    "models/embedding-002": "all-mpnet-base-v2",  # Add this
}
```

### Currently Registered:
- `models/embedding-001` → `all-MiniLM-L6-v2` (384 dimensions)
- `models/all-MiniLM-L6-v2` → `all-MiniLM-L6-v2` (384 dimensions)

### Popular Sentence-Transformer Models:

| Model | Dimensions | Max Seq | Size |
|-------|------------|---------|------|
| all-MiniLM-L6-v2 | 384 | 256 | ~80MB |
| all-mpnet-base-v2 | 768 | 512 | ~420MB |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 512 | ~470MB |

## Model Caching

- Models stored in `models/` directory
- Downloaded once, reused across restarts
- Excluded from git via `.gitignore`
- ~80-500MB per model

## Testing

```bash
# Using curl
curl http://localhost:8800/health

curl -X POST http://localhost:8800/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "models/embedding-001", "texts": ["Hello, world!"]}'

# Using test script
python test_service.py
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make start` | Start with PM2 |
| `make stop` | Stop service |
| `make restart` | Restart service |
| `make status` | Show status |
| `make logs` | Show logs |
| `make monit` | Real-time monitoring |
| `make uninstall` | Remove from PM2 |
| `make clean` | Clean logs |

## Configuration

Edit `ecosystem.config.js` to change:
- Port (default: 8000)
- Memory limit
- Log locations
- Instance count

## Troubleshooting

**Model not loading:**
```bash
make logs  # Check for errors
# Verify internet connection on first run
# Check disk space (80-500MB per model)
```

**Port already in use:**
```bash
# Change port in ecosystem.config.js
# Or stop the conflicting service
```

**Service not starting:**
```bash
pm2 logs embedding-service  # Check PM2 logs
make install  # Verify dependencies
```

## Default Model Info

- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Max Sequence**: 256 tokens
- **Normalization**: L2 normalized
- **Storage**: `models/` directory
