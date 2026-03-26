from groq import Groq
from src.config import settings


# Single Groq client — instantiated once at module level
_client = Groq(api_key=settings.GROQ_API_KEY)


def generate_answer(question: str, contexts: list[dict], model_key: str) -> str:
    """
    Generate an answer using Groq LLM based ONLY on retrieved contexts.

    Args:
        question:   The user's question
        contexts:   List of dicts from retriever, each has a "chunk" key
        model_key:  Key from settings.GROQ_MODELS (e.g. "llama-3.1-8b  Fast")

    Returns:
        The LLM's answer as a string
    """
    if not contexts:
        return "Not found in the document!"

    # Build context block from retrieved chunks
    context_text = "\n\n".join(c["chunk"] for c in contexts)

    prompt = f"""You are a helpful assistant answering ONLY from the given context.
    If the answer is not present in the context, say exactly: "Not found in the document!".
    Do not use any external knowledge.

    Context:
    {context_text}

    Question: {question}
    Answer:"""

    # Resolve display name → real Groq model ID
    model_id = settings.GROQ_MODELS[model_key]

    response = _client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()
