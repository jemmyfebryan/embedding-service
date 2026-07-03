from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import os


class EmbeddingService:
    """Service for handling text embeddings using sentence-transformers with dynamic model loading"""

    # Model registry: maps API model names to sentence-transformer model names
    MODEL_REGISTRY = {
        "models/embedding-001": "all-MiniLM-L6-v2",
        "models/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
        # Add more models here, e.g.:
        # "models/embedding-002": "all-mpnet-base-v2",
        # "models/embedding-003": "paraphrase-multilingual-MiniLM-L12-v2",
    }

    def __init__(self, cache_dir: str = "models"):
        """
        Initialize the embedding service with model caching

        Args:
            cache_dir: Directory to store downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.models: Dict[str, SentenceTransformer] = {}
        self.model_info_cache: Dict[str, dict] = {}

    def _resolve_model_name(self, api_model_name: str) -> str:
        """
        Resolve API model name to sentence-transformer model name

        Args:
            api_model_name: Model name from API request

        Returns:
            Actual sentence-transformer model name
        """
        if api_model_name in self.MODEL_REGISTRY:
            return self.MODEL_REGISTRY[api_model_name]

        # If not in registry, assume it's already a sentence-transformer model name
        return api_model_name

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a model with local caching

        Args:
            model_name: Name of the sentence-transformer model

        Returns:
            Loaded SentenceTransformer model
        """
        # Check if model is already loaded
        if model_name in self.models:
            return self.models[model_name]

        # Load model with local cache
        try:
            print(f"Loading model: {model_name}")
            print(f"Cache directory: {self.cache_dir.absolute()}")

            model = SentenceTransformer(
                model_name,
                cache_folder=str(self.cache_dir)
            )

            self.models[model_name] = model
            print(f"Successfully loaded model: {model_name}")

            # Cache model info
            self.model_info_cache[model_name] = {
                "model_name": model_name,
                "max_seq_length": model.max_seq_length,
                "embedding_dimension": model.get_sentence_embedding_dimension(),
            }

            return model

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def encode(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
        """
        Encode texts into embeddings using specified model

        Args:
            texts: List of text strings to encode
            model_name: Model to use for encoding

        Returns:
            List of embedding vectors
        """
        model = self._load_model(model_name)

        # Generate embeddings
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Convert to list format
        return embeddings.tolist()

    def is_ready(self, model_name: Optional[str] = None) -> bool:
        """
        Check if a model is loaded and ready

        Args:
            model_name: Optional model name to check. If None, checks if any model is loaded.

        Returns:
            True if model is ready
        """
        if model_name:
            return model_name in self.models
        return len(self.models) > 0

    def get_model_info(self, model_name: Optional[str] = None) -> dict:
        """
        Get information about loaded models

        Args:
            model_name: Optional model name to get info for. If None, returns info for all loaded models.

        Returns:
            Dictionary with model information
        """
        if model_name:
            if model_name in self.model_info_cache:
                return self.model_info_cache[model_name]
            return {"error": f"Model {model_name} not loaded"}

        return {
            "loaded_models": list(self.models.keys()),
            "model_info": self.model_info_cache,
            "available_models": list(self.MODEL_REGISTRY.keys())
        }

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names"""
        return list(self.models.keys())

    def resolve_api_model(self, api_model: str) -> str:
        """
        Resolve API model identifier to sentence-transformer model name

        Args:
            api_model: API model identifier (e.g., "models/embedding-001")

        Returns:
            Sentence-transformer model name
        """
        return self._resolve_model_name(api_model)
