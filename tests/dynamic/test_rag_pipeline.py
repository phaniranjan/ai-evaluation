"""Dynamic RAG pipeline evaluation tests using DeepEval metrics."""

import json
import logging
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from ai_evaluation.rag_pipeline import RAGPipeline
from ai_evaluation.rag_retriever import BM25Retriever

logger = logging.getLogger(__name__)


@pytest.fixture
def rag_pipeline(llm_generator):
    """Fixture initializing BM25Retriever with news articles and returning RAGPipeline."""
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

    return RAGPipeline(retriever=retriever, llm_generator=llm_generator, default_top_k=2)


@pytest.mark.dynamic
def test_rag_pipeline_eval(judge_model, rag_pipeline):
    """Evaluate RAG pipeline using Faithfulness, Contextual Relevancy, and Answer Relevancy."""
    query = (
        "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve, "
        "and under what test conditions was its environmental stability demonstrated?"
    )

    logger.info("Executing RAG pipeline evaluation query: %s", query)
    result = rag_pipeline.query(query, top_k=2)

    logger.info(
        "Retrieved Context Chunks (%d): %s",
        len(result["retrieval_context"]),
        result["retrieval_context"],
    )
    logger.info("Generated RAG Answer:\n%s", result["answer"])

    # Ensure retrieval returned context
    assert (
        len(result["retrieval_context"]) > 0
    ), "Retrieval context must not be empty for in-context query"

    # Define DeepEval metrics
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7, model=judge_model)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    test_case = LLMTestCase(
        input=result["query"],
        actual_output=result["answer"],
        retrieval_context=result["retrieval_context"],
    )

    logger.info(
        "Asserting DeepEval metrics (Faithfulness, Contextual Relevancy, Answer Relevancy)..."
    )
    assert_test(
        test_case,
        [faithfulness_metric, contextual_relevancy_metric, answer_relevancy_metric],
    )
    logger.info("RAG pipeline dynamic evaluation passed successfully!")


@pytest.mark.dynamic
def test_rag_out_of_context_negative(rag_pipeline):
    """Negative test: query answer not in news articles must state information is not provided."""
    query = "What was the final score of the 2026 NFL Super Bowl?"

    logger.info("Executing out-of-context negative test query: %s", query)
    result = rag_pipeline.query(query, top_k=2)

    logger.info("Out-of-context retrieval context: %s", result["retrieval_context"])
    logger.info("Out-of-context generated answer:\n%s", result["answer"])

    # Verify that Gemini grounded generator explicitly states the information is unavailable
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
    logger.info("Out-of-context negative test passed!")


@pytest.mark.dynamic
def test_wrong_retrieval_contextual_relevancy_degradation(judge_model):
    """Adversarial test: forcing wrong context (JWST content for a solar query) degrades ContextualRelevancyMetric."""
    query = "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve?"
    # Forced wrong context (JWST astronomy article instead of solar cell article)
    wrong_context = [
        "NASA's James Webb Space Telescope (JWST) has detected carbon-bearing molecules, including methane and carbon dioxide, in the atmosphere of exoplanet K2-18b located 120 light-years from Earth in Leo."
    ]
    actual_output = "The solar cell achieved 34.6% power conversion efficiency."

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=wrong_context,
    )

    metric = ContextualRelevancyMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "Wrong retrieval ContextualRelevancyMetric score: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )
    assert (
        metric.score < 0.7
    ), f"Expected ContextualRelevancyMetric to fail (< 0.7), but got score: {metric.score}"
    logger.info("Wrong retrieval contextual relevancy degradation verified successfully!")


@pytest.mark.dynamic
def test_hallucinated_answer_faithfulness_degradation(judge_model):
    """Adversarial test: an answer with fabricated/hallucinated facts degrades FaithfulnessMetric."""
    query = "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve?"
    correct_context = [
        "Researchers at the Helmholtz-Zentrum Berlin have developed a novel perovskite-silicon tandem solar cell that achieved a certified power conversion efficiency of 34.6% in laboratory tests."
    ]
    # Fabricated answer claiming facts absent from the context
    hallucinated_output = (
        "The perovskite-silicon tandem solar cell achieved 99.9% power conversion efficiency, "
        "was invented by Elon Musk in 2012, and runs on lunar gravity."
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=hallucinated_output,
        retrieval_context=correct_context,
    )

    metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "Hallucinated answer FaithfulnessMetric score: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )
    assert (
        metric.score < 0.7
    ), f"Expected FaithfulnessMetric to fail (< 0.7), but got score: {metric.score}"
    logger.info("Hallucinated answer faithfulness degradation verified successfully!")


@pytest.mark.dynamic
def test_irrelevant_answer_relevancy_degradation(judge_model):
    """Adversarial test: an unhelpful/irrelevant response degrades AnswerRelevancyMetric."""
    query = "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve?"
    correct_context = [
        "Researchers at the Helmholtz-Zentrum Berlin have developed a novel perovskite-silicon tandem solar cell that achieved a certified power conversion efficiency of 34.6% in laboratory tests."
    ]
    # Irrelevant response dodging the question about efficiency
    irrelevant_output = "Solar panels are generally rectangular, blue or black, and mounted on residential rooftops during bright sunny weather."

    test_case = LLMTestCase(
        input=query,
        actual_output=irrelevant_output,
        retrieval_context=correct_context,
    )

    metric = AnswerRelevancyMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "Irrelevant answer AnswerRelevancyMetric score: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )
    assert (
        metric.score < 0.7
    ), f"Expected AnswerRelevancyMetric to fail (< 0.7), but got score: {metric.score}"
    logger.info("Irrelevant answer relevancy degradation verified successfully!")
