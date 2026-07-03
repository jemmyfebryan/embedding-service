# Embedding Service

A local embedding service powered by FastAPI and `sentence-transformers` with dynamic model loading support. The API structure mimics Google's Gemini embedding API for easy integration and seamless swapping between local and cloud services.

## Features

- FastAPI-based REST API
- **Dynamic model loading** - Models loaded on-demand from API requests
- **Local model caching** - Models stored in `models/` directory, no re-downloading needed
- **Multi-model support** - Easy to add new embedding models
- Gemini-compatible API structure
- CPU-only PyTorch (lightweight, no CUDA dependencies)
- Single text embedding endpoint
- Batch embedding endpoint
- Health check and model listing endpoints
- PM2 process management

## Quick Start

### 1. Install Dependencies

```bash
make install
```

Or manually:
```bash
pip install -r requirements.txt
```

### 2. Start the Service

```bash
make start
```

Or manually:
```bash
pm2 start ecosystem.config.js
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
```bash
# Open in browser
http://localhost:8000/docs
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

### Single Text Embedding (Gemini-compatible)
```
POST /v1/models/{model_name}:embedText
```

Request:
```json
{
  "model": "models/embedding-001",
  "content": {
    "parts": ["Your text string here"]
  }
}
```

Response:
```json
{
  "embedding": {
    "embeddings": [
      {
        "value": {
          "values": [0.1, -0.2, 0.3, ...]
        }
      }
    ]
  }
}
```

### Batch Embedding (Gemini-compatible)
```
POST /v1/models/{model_name}:batchEmbedText
```

Request:
```json
{
  "model": "models/embedding-001",
  "contents": [
    {"parts": ["First text"]},
    {"parts": ["Second text"]}
  ]
}
```

## Dynamic Model Selection

Models can be specified in three ways:

### 1. Via URL Path
```bash
POST /v1/models/all-MiniLM-L6-v2:embedText
{
  "content": {"parts": ["Text to embed"]}
}
```

### 2. Via Request Body
```bash
POST /v1/models/embedding-001:embedText
{
  "model": "models/all-MiniLM-L6-v2",
  "content": {"parts": ["Text to embed"]}
}
```

### 3. Using Registered Model Names
Currently registered models (see `app/embedding.py` to add more):

- `models/embedding-001` → `all-MiniLM-L6-v2` (default)
- `models/all-MiniLM-L6-v2` → `all-MiniLM-L6-v2`

## Adding New Models

Edit `app/embedding.py` and add to the `MODEL_REGISTRY`:

```python
MODEL_REGISTRY = {
    "models/embedding-001": "all-MiniLM-L6-v2",
    "models/embedding-002": "all-mpnet-base-v2",  # Add this
    "models/embedding-003": "paraphrase-multilingual-MiniLM-L12-v2",  # Add this
}
```

The model will be automatically downloaded and cached in the `models/` directory on first use.

## Model Caching

- Models are stored in the `models/` directory
- Downloaded once, reused across restarts
- No need to re-download after service restart
- Excluded from git via `.gitignore`

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make start` | Start the service with PM2 |
| `make stop` | Stop the service |
| `make restart` | Restart the service |
| `make status` | Show service status |
| `make logs` | Show service logs |
| `make monit` | Monitor service in real-time |
| `make uninstall` | Remove from PM2 |
| `make clean` | Clean logs |

## Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Single embedding
curl -X POST http://localhost:8000/v1/models/embedding-001:embedText \
  -H "Content-Type: application/json" \
  -d '{"content": {"parts": ["Hello, world!"]}}'

# Batch embedding
curl -X POST http://localhost:8000/v1/models/embedding-001:batchEmbedText \
  -H "Content-Type: application/json" \
  -d '{"contents": [{"parts": ["First text"]}, {"parts": ["Second text"]}]}'
```

### Using the test script

```bash
python test_service.py
```

The test script will:
- Test all endpoints
- Test dynamic model selection
- Verify embedding consistency
- Check the models directory

## Configuration

Edit `ecosystem.config.js` to change:
- Port (default: 8000)
- Memory limit
- Log file locations
- Number of instances

## Supported Models

The service uses sentence-transformers and supports any model from their repository. Popular options:

| Model | Dimensions | Max Seq Length | Size |
|-------|------------|---------------|------|
| all-MiniLM-L6-v2 | 384 | 256 | ~80MB |
| all-mpnet-base-v2 | 768 | 512 | ~420MB |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 512 | ~470MB |

## Model Information

Default model (all-MiniLM-L6-v2):
- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Max Sequence Length**: 256 tokens
- **Normalization**: L2 (enabled by default)
- **Storage**: `models/` directory

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

### Model not loading
- Check logs: `make logs`
- Verify internet connection on first run (model download)
- Check disk space (models need ~80-500MB each)

### Port already in use
- Change port in `ecosystem.config.js`
- Or stop the conflicting service

### Service not starting
- Check PM2 logs: `pm2 logs embedding-service`
- Verify Python dependencies installed
- Check for syntax errors in code
