#!/usr/bin/env python3
"""Test script for the embedding service with dynamic model support"""

import requests
import json


BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint"""
    print("=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    print()


def test_health():
    """Test health endpoint"""
    print("=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_list_models():
    """Test models list endpoint"""
    print("=== Testing Models List Endpoint ===")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Available models: {result.get('available_models', [])}")
    print(f"Loaded models: {result.get('loaded_models', [])}")
    print(f"Model info: {json.dumps(result.get('model_info', {}), indent=2)}")
    print()


def test_single_embedding():
    """Test single text embedding"""
    print("=== Testing Single Text Embedding ===")
    payload = {
        "model": "models/embedding-001",
        "content": {
            "parts": ["Hello, world! This is a test."]
        }
    }
    response = requests.post(
        f"{BASE_URL}/v1/models/embedding-001:embedText",
        json=payload
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Embedding dimension: {len(result['embedding']['embeddings'][0]['value']['values'])}")
    print(f"First 10 values: {result['embedding']['embeddings'][0]['value']['values'][:10]}")
    print()


def test_batch_embedding():
    """Test batch embedding"""
    print("=== Testing Batch Embedding ===")
    payload = {
        "model": "models/embedding-001",
        "contents": [
            {"parts": ["First text string"]},
            {"parts": ["Second text string"]},
            {"parts": ["Third text string"]}
        ]
    }
    response = requests.post(
        f"{BASE_URL}/v1/models/embedding-001:batchEmbedText",
        json=payload
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Number of embeddings: {len(result['embedding']['embeddings'])}")
    print(f"Embedding dimension: {len(result['embedding']['embeddings'][0]['value']['values'])}")
    print()


def test_dynamic_model():
    """Test dynamic model selection via URL"""
    print("=== Testing Dynamic Model Selection (URL) ===")
    payload = {
        "content": {
            "parts": ["Testing dynamic model via URL path"]
        }
    }
    # Use all-MiniLM-L6-v2 directly in the URL
    response = requests.post(
        f"{BASE_URL}/v1/models/all-MiniLM-L6-v2:embedText",
        json=payload
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Embedding dimension: {len(result['embedding']['embeddings'][0]['value']['values'])}")
    else:
        print(f"Error: {response.text}")
    print()


def test_model_override():
    """Test model override in request body"""
    print("=== Testing Model Override in Request Body ===")
    payload = {
        "model": "models/all-MiniLM-L6-v2",
        "content": {
            "parts": ["Testing model override in request body"]
        }
    }
    response = requests.post(
        f"{BASE_URL}/v1/models/embedding-001:embedText",
        json=payload
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Embedding dimension: {len(result['embedding']['embeddings'][0]['value']['values'])}")
    else:
        print(f"Error: {response.text}")
    print()


def test_consistency():
    """Test that same text produces same embedding"""
    print("=== Testing Embedding Consistency ===")
    text = "Consistency test text"
    payload = {
        "model": "models/embedding-001",
        "content": {
            "parts": [text]
        }
    }

    response1 = requests.post(
        f"{BASE_URL}/v1/models/embedding-001:embedText",
        json=payload
    )
    response2 = requests.post(
        f"{BASE_URL}/v1/models/embedding-001:embedText",
        json=payload
    )

    if response1.status_code == 200 and response2.status_code == 200:
        emb1 = response1.json()['embedding']['embeddings'][0]['value']['values']
        emb2 = response2.json()['embedding']['embeddings'][0]['value']['values']

        # Check if embeddings are identical
        are_identical = emb1 == emb2
        print(f"Embeddings are identical: {are_identical}")

        # Calculate cosine similarity (should be 1.0 for identical vectors)
        import numpy as np
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"Cosine similarity: {similarity:.6f}")
    else:
        print("Error: One or both requests failed")
    print()


if __name__ == "__main__":
    try:
        test_root()
        test_health()
        test_list_models()
        test_single_embedding()
        test_batch_embedding()
        test_dynamic_model()
        test_model_override()
        test_consistency()
        print("✅ All tests passed!")

        # Check models directory
        print("\n=== Checking Models Directory ===")
        import os
        if os.path.exists("models"):
            print("Models directory exists!")
            print(f"Directory contents: {os.listdir('models')}")
        else:
            print("Models directory not found (will be created on first use)")

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the service. Is it running?")
        print("Start it with: make start")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
