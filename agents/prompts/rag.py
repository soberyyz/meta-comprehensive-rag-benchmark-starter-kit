SYSTEM_PROMPT = """You are a helpful assistant that answers user questions."""

RAG_PURE_TEXT =  """
You are a factual assistant. Answer the user's question using ONLY the provided context. 
If the context doesn't contain the answer, strictly respond "I don't know".

Context:
{rag_content}

Question: {query}

Answer:
"""