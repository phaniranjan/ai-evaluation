"""RAG Metric Validation tests verifying DeepEval evaluator sensitivity to failure modes."""

import logging

import pytest
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_contextual_relevancy_metric_validation(judge_model):
    """Metric Validation: Injecting wrong context (JWST content for solar query) degrades ContextualRelevancyMetric."""
    query = "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve?"
    # Forced wrong context (JWST astronomy article chunk)
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
    logger.info("ContextualRelevancyMetric validation verified successfully!")


@pytest.mark.dynamic
def test_faithfulness_metric_validation(judge_model):
    """Metric Validation: Injecting an answer with fabricated/hallucinated facts degrades FaithfulnessMetric."""
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
    logger.info("FaithfulnessMetric validation verified successfully!")


@pytest.mark.dynamic
def test_answer_relevancy_metric_validation(judge_model):
    """Metric Validation: Injecting an unhelpful/irrelevant response degrades AnswerRelevancyMetric."""
    query = "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve?"
    correct_context = [
        "Researchers at the Helmholtz-Zentrum Berlin have developed a novel perovskite-silicon tandem solar cell that achieved a certified power conversion efficiency of 34.6% in laboratory tests."
    ]
    # Irrelevant response dodging the query about efficiency
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
    logger.info("AnswerRelevancyMetric validation verified successfully!")
