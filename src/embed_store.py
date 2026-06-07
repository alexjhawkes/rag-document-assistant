import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_embeddings(chunks: list) -> list:
    """
    Generate an embedding for each text chunk using OpenAI.

    Args:
        chunks: List of text chunk strings

    Returns:
        List of embeddings (each embedding is a list of numbers)
    """
    embeddings = []

    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
        print(f"Embedded chunk {i+1} of {len(chunks)}")

    return embeddings


def store_in_faiss(embeddings: list, chunks: list, index_path: str = "data/faiss_index", chunks_path: str = "data/chunks.json"):
    """
    Store embeddings in a FAISS index and save chunks to disk.

    Args:
        embeddings: List of embedding vectors
        chunks: List of original text chunks
        index_path: Where to save the FAISS index file
        chunks_path: Where to save the chunks as JSON
    """
    # Convert embeddings to a numpy array (FAISS requires this format)
    vectors = np.array(embeddings, dtype="float32")

    # Get the number of dimensions (1536 for text-embedding-3-small)
    dimension = vectors.shape[1]

    # Create a FAISS index
    # IndexFlatL2 means: store all vectors and search by L2 distance (similarity)
    index = faiss.IndexFlatL2(dimension)

    # Add our vectors to the index
    index.add(vectors)

    # Save the FAISS index to disk
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")

    # Save the original chunks to disk as JSON
    # We need these to return the actual text when a match is found
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved to: {chunks_path}")

    return index


def main():
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from extract_text import extract_text_from_pdf
    from chunk_text import clean_text, chunk_text

    pdf_path = Path("data/sample_pdfs/my_document.pdf")
    raw_text = extract_text_from_pdf(str(pdf_path))
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, chunk_size=100, overlap=20)

    print(f"Generating embeddings for {len(chunks)} chunks...\n")
    embeddings = generate_embeddings(chunks)

    print(f"\nStoring in FAISS...\n")
    store_in_faiss(embeddings, chunks)

    print(f"\nDone! {len(chunks)} chunks indexed and ready to search.")


if __name__ == "__main__":
    main()
