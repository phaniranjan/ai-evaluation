"""RAG Pipeline combining BM25 retriever and Gemini LLM generator with strict grounding."""

import logging
from typing import Any, Dict, List, Optional

from ai_evaluation.llm_generator import LLMGenerator
from ai_evaluation.rag_retriever import BM25Retriever

logger = logging.getLogger(__name__)

STRICT_GROUNDING_INSTRUCTION = (
    "Answer the user's question strictly using ONLY the provided context. "
    "If the answer cannot be found in the context, explicitly state that the information is not provided."
)


class RAGPipeline:
    """RAG pipeline integrating BM25 chunk retrieval with grounded Gemini generation."""

    def __init__(
        self,
        retriever: BM25Retriever,
        llm_generator: LLMGenerator,
        default_top_k: int = 2,
        system_instruction: str = STRICT_GROUNDING_INSTRUCTION,
    ) -> None:
        self.retriever = retriever
        self.llm_generator = llm_generator
        self.default_top_k = default_top_k
        self.system_instruction = system_instruction

    def query(self, user_query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Execute RAG pipeline: retrieve context, construct prompt, and generate answer.

        Args:
            user_query: Question asked by the user.
            top_k: Optional override for number of chunks to retrieve.

        Returns:
            Dict containing 'query', 'answer', 'retrieval_context', and 'retrieved_chunks'.
        """
        k = top_k if top_k is not None else self.default_top_k
        logger.info("Executing RAG query (top_k=%d): %s", k, user_query)

        # Retrieve relevant chunks with score & metadata
        retrieved_chunks = self.retriever.retrieve_chunks(user_query, top_k=k)
        # DeepEval retrieval_context expects text strings of non-zero scoring chunks
        retrieval_context = [c["text"] for c in retrieved_chunks if c["score"] > 0.0]

        if retrieval_context:
            context_block = "\n\n---\n\n".join(retrieval_context)
        else:
            context_block = "No relevant context found in documents."

        formatted_prompt = (
            f"Context Information:\n{context_block}\n\n" f"User Question: {user_query}"
        )

        logger.debug("Generating grounded response with Gemini LLMGenerator...")
        answer = self.llm_generator.generate_response(
            question=formatted_prompt,
            system_instruction=self.system_instruction,
        )

        return {
            "query": user_query,
            "answer": answer,
            "retrieval_context": retrieval_context,
            "retrieved_chunks": retrieved_chunks,
        }
