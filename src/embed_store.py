import os
from openai import OpenAI
from dotenv import load_dotenv

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


def main():
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from pathlib import Path
    from extract_text import extract_text_from_pdf
    from chunk_text import clean_text, chunk_text

    pdf_path = Path("data/sample_pdfs/my_document.pdf")
    raw_text = extract_text_from_pdf(str(pdf_path))
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, chunk_size=100, overlap=20)

    print(f"Generating embeddings for {len(chunks)} chunks...\n")
    embeddings = generate_embeddings(chunks)

    print(f"\nDone! Each embedding has {len(embeddings[0])} numbers")
    print(f"First 5 numbers of embedding 1: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()
