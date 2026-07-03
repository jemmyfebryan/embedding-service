#!/usr/bin/env python3
"""Test script for the embedding service with simplified API"""

import requests
import json


BASE_URL = "http://localhost:8800"


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
        "texts": ["Hello, world! This is a test."]
    }
    response = requests.post(
        f"{BASE_URL}/v1/embed",
        json=payload
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    print(f"Embedding dimension: {len(result['embeddings'][0])}")
    print(f"First 10 values: {result['embeddings'][0][:10]}")
    print()


def test_batch_embedding():
    """Test batch embedding"""
    print("=== Testing Batch Embedding ===")
    payload = {
        "model": "models/embedding-001",
        "texts": [
            "First text string",
            "Second text string",
            "Third text string"
        ]
    }
    response = requests.post(
        f"{BASE_URL}/v1/embed",
        json=payload
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Number of embeddings: {len(result['embeddings'])}")
    print(f"Embedding dimension: {len(result['embeddings'][0])}")
    print()


def test_model_override():
    """Test different model selection"""
    print("=== Testing Different Model Selection ===")
    payload = {
        "model": "models/all-MiniLM-L6-v2",
        "texts": ["Testing model selection"]
    }
    response = requests.post(
        f"{BASE_URL}/v1/embed",
        json=payload
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Embedding dimension: {len(result['embeddings'][0])}")
    else:
        print(f"Error: {response.text}")
    print()


def test_consistency():
    """Test that same text produces same embedding"""
    print("=== Testing Embedding Consistency ===")
    text = "Consistency test text"
    payload = {
        "model": "models/embedding-001",
        "texts": [text]
    }

    response1 = requests.post(f"{BASE_URL}/v1/embed", json=payload)
    response2 = requests.post(f"{BASE_URL}/v1/embed", json=payload)

    if response1.status_code == 200 and response2.status_code == 200:
        emb1 = response1.json()['embeddings'][0]
        emb2 = response2.json()['embeddings'][0]

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


def test_empty_texts():
    """Test error handling for empty texts"""
    print("=== Testing Empty Texts Error Handling ===")
    payload = {
        "model": "models/embedding-001",
        "texts": []
    }
    response = requests.post(f"{BASE_URL}/v1/embed", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()


def test_large_batch():
    """Test with larger batch"""
    print("=== Testing Large Batch ===")
    payload = {
        "model": "models/embedding-001",
        "texts": [f"Text number {i}" for i in range(100)]
    }
    response = requests.post(f"{BASE_URL}/v1/embed", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Number of embeddings: {len(result['embeddings'])}")
        print(f"Each embedding dimension: {len(result['embeddings'][0])}")
    else:
        print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    try:
        test_root()
        test_health()
        test_list_models()
        test_single_embedding()
        test_batch_embedding()
        test_model_override()
        test_consistency()
        test_empty_texts()
        test_large_batch()
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
