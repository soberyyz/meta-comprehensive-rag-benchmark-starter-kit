RERWRITE_QUERY_BY_IMAGE = """
You need to rewrite the user's question by incorporating details from their uploaded image. 
Follow these steps:
1. Identify which aspects of the question need clarification based on the image content.
2. Rewrite the question to:
   - Be more specific and actionable
   - Include relevant details from the image description
   - Maintain the original intent of the question

The question: {query}
The rewritten question:"""