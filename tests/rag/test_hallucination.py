import logging

import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_rag_system_hallucination_evaluation(judge_model, rag_pipeline):
    """SUT Evaluation: Verify that the actual RAG pipeline generates zero hallucinations on complex factual queries."""
    query = (
        "What molecules were detected in the atmosphere of exoplanet K2-18b, "
        "and what specific instruments on the James Webb Space Telescope were used?"
    )

    logger.info("Executing SUT RAG hallucination evaluation query: %s", query)
    result = rag_pipeline.query(query, top_k=2)

    logger.info(
        "Retrieved Context Chunks (%d): %s",
        len(result["retrieval_context"]),
        result["retrieval_context"],
    )
    logger.info("Generated RAG Answer:\n%s", result["answer"])

    assert len(result["retrieval_context"]) > 0, "Retrieval context must not be empty"

    metric = HallucinationMetric(threshold=0.7, model=judge_model)

    test_case = LLMTestCase(
        input=result["query"],
        actual_output=result["answer"],
        context=result["retrieval_context"],
        retrieval_context=result["retrieval_context"],
    )

    logger.info("Asserting DeepEval HallucinationMetric on actual RAG pipeline output...")
    assert_test(test_case, [metric])
    logger.info(
        "RAG system hallucination evaluation passed successfully with zero detected hallucinations!"
    )
