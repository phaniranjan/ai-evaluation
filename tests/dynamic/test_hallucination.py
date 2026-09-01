"""System Under Test (SUT) evaluation testing RAG pipeline for hallucinations."""

import json
import logging
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from ai_evaluation.rag_pipeline import RAGPipeline
from ai_evaluation.rag_retriever import BM25Retriever

logger = logging.getLogger(__name__)


@pytest.fixture
def rag_retriever():
    """Fixture initializing BM25Retriever loaded with news articles."""
    data_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ai_evaluation"
        / "data"
        / "rag_news_articles.json"
    )
    with data_path.open(encoding="utf-8") as f:
        articles = json.load(f)

    retriever = BM25Retriever()
    retriever.add_articles(articles)
    return retriever


@pytest.fixture
def rag_pipeline(llm_generator, rag_retriever):
    """Fixture returning RAGPipeline backed by BM25Retriever and LLMGenerator."""
    return RAGPipeline(retriever=rag_retriever, llm_generator=llm_generator, default_top_k=2)


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
