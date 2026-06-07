import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query: str, chunks: list) -> str:
    """
    Send retrieved chunks and a question to the LLM to generate an answer.

    Args:
        query: The user's question
        chunks: List of relevant text chunks retrieved from FAISS

    Returns:
        The LLM's answer as a string
    """
    # Combine the chunks into one block of context
    context = "\n\n".join([result["chunk"] for result in chunks])

    # Build the prompt
    prompt = f"""You are a helpful assistant. Answer the question below using only the context provided.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


def main():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from retrieve import retrieve_chunks

    # Try a few different questions
    questions = [
        "What is Alex's work experience?",
        "What technologies does Alex know?",
        "What is Alex's highest level of education?"
    ]

    for query in questions:
        print(f"\nQuestion: {query}")
        print("-" * 50)
        chunks = retrieve_chunks(query, top_k=3)
        answer = generate_answer(query, chunks)
        print(f"Answer: {answer}\n")
        print("=" * 50)


if __name__ == "__main__":
    main()
