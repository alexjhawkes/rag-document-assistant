import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_faiss_index(index_path: str = "data/faiss_index", chunks_path: str = "data/chunks.json"):
    """
    Load the FAISS index and chunks from disk.

    Returns:
        index: The FAISS index
        chunks: The list of original text chunks
    """
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def embed_query(query: str) -> np.ndarray:
    """
    Convert a question into an embedding vector.

    Args:
        query: The user's question as a string

    Returns:
        A numpy array of the query embedding
    """
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    return np.array([embedding], dtype="float32")


def retrieve_chunks(query: str, top_k: int = 3) -> list:
    """
    Find the most relevant chunks for a given query.

    Args:
        query: The user's question
        top_k: How many chunks to return

    Returns:
        List of the most relevant text chunks
    """
    index, chunks = load_faiss_index()
    query_vector = embed_query(query)

    # Search FAISS for the top_k closest vectors
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "distance": float(distances[0][i])
        })

    return results


def main():
    query = "What is Alex's work experience?"

    print(f"Question: {query}\n")
    print("Searching for relevant chunks...\n")

    results = retrieve_chunks(query, top_k=3)

    for i, result in enumerate(results):
        print(f"--- Result {i+1} (distance: {result['distance']:.4f}) ---")
        print(result["chunk"])
        print()


if __name__ == "__main__":
    main()
