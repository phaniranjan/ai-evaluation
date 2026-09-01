import logging

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_rag_retrieval_and_top_k_behavior(rag_retriever):
    """Retrieval System Test: Verify BM25 chunk metadata, relevance scoring, and top-k behavior."""
    query = "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve?"

    # Test top_k=1
    top1_chunks = rag_retriever.retrieve_chunks(query, top_k=1)
    assert len(top1_chunks) == 1, "top_k=1 must return exactly 1 chunk"
    assert (
        top1_chunks[0]["article_id"] == "art_solar_2026"
    ), "Top retrieved chunk must belong to solar cell article"
    assert top1_chunks[0]["score"] > 0.0, "Top chunk BM25 score must be positive"
    assert (
        "34.6%" in top1_chunks[0]["text"]
    ), "Top retrieved chunk must contain efficiency metric 34.6%"

    # Test top_k=2
    top2_chunks = rag_retriever.retrieve_chunks(query, top_k=2)
    assert len(top2_chunks) == 2, "top_k=2 must return 2 chunks"
    assert (
        top2_chunks[0]["score"] >= top2_chunks[1]["score"]
    ), "Chunks must be ordered descending by BM25 score"

    # Test top_k=3
    top3_chunks = rag_retriever.retrieve_chunks(query, top_k=3)
    assert len(top3_chunks) == 3, "top_k=3 must return 3 chunks"

    logger.info("Retrieval top-k behavior and chunk precision verified successfully!")


@pytest.mark.dynamic
def test_rag_generation_grounding(rag_pipeline):
    """Generation System Test: Verify generated answer is grounded in retrieved context."""
    query = (
        "What is the name of the personalized cancer vaccine and what risk reduction did it show?"
    )

    result = rag_pipeline.query(query, top_k=2)

    logger.info("Retrieved context chunks for mRNA vaccine query: %s", result["retrieval_context"])
    logger.info("Generated response:\n%s", result["answer"])

    assert len(result["retrieval_context"]) > 0, "Retrieval context must not be empty"
    answer_text = result["answer"]

    assert "mRNA-4157" in answer_text, "Answer must mention vaccine name mRNA-4157 from context"
    assert "44%" in answer_text, "Answer must cite 44% risk reduction from context"
    logger.info("Generation grounding test passed successfully!")


@pytest.mark.dynamic
def test_rag_out_of_context_refusal(rag_pipeline):
    """System Refusal Test: Verify out-of-context queries generate an explicit disclaimer."""
    query = "What was the final score of the 2026 NFL Super Bowl?"

    result = rag_pipeline.query(query, top_k=2)

    logger.info("Out-of-context generated answer:\n%s", result["answer"])

    answer_lower = result["answer"].lower()
    disclaimer_phrases = [
        "not provided",
        "not available",
        "no information",
        "cannot be found",
        "does not contain",
        "no relevant context",
        "is not mentioned",
    ]
    has_disclaimer = any(phrase in answer_lower for phrase in disclaimer_phrases)

    assert (
        has_disclaimer
    ), f"Generated response for out-of-context query did not state unavailability. Answer: {result['answer']}"
    logger.info("System out-of-context refusal test passed successfully!")


@pytest.mark.dynamic
def test_rag_pipeline_end_to_end(judge_model, rag_pipeline):
    """End-to-End System Evaluation: Evaluate full RAG query using Faithfulness, Contextual Relevancy, and Answer Relevancy."""
    query = (
        "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve, "
        "and under what test conditions was its environmental stability demonstrated?"
    )

    result = rag_pipeline.query(query, top_k=2)

    logger.info("End-to-End Generated RAG Answer:\n%s", result["answer"])

    assert len(result["retrieval_context"]) > 0, "Retrieval context must not be empty"

    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7, model=judge_model)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    test_case = LLMTestCase(
        input=result["query"],
        actual_output=result["answer"],
        retrieval_context=result["retrieval_context"],
    )

    logger.info("Asserting DeepEval end-to-end metrics...")
    assert_test(
        test_case,
        [faithfulness_metric, contextual_relevancy_metric, answer_relevancy_metric],
    )
    logger.info("End-to-end RAG system evaluation passed successfully!")
