RAG_SYNTHESIS_INSTRUCTION = """
## Document Synthesis Guidelines:
You have received excerpts from internal documentation. Use them to answer the user's question:
1. Read and carefully understand the content of each excerpt.
2. Synthesize the information into a coherent response without mechanical copy-pasting.
3. If information is insufficient, inform the user clearly.
4. Cite sources (filename) at the end of your response.

All user-facing responses must be in Vietnamese with proper diacritics (tiếng Việt có dấu đầy đủ).
"""
