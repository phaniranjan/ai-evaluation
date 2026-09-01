"""Evaluator Validation tests for DeepEval HallucinationMetric."""

import logging

import pytest
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_hallucination_metric_detects_hallucination(judge_model):
    """Evaluator Validation: Verify HallucinationMetric detects ungrounded fabricated claims and drops score."""
    query = "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve?"
    context = [
        "Researchers at the Helmholtz-Zentrum Berlin have developed a novel perovskite-silicon tandem solar cell that achieved a certified power conversion efficiency of 34.6% in laboratory tests."
    ]
    # Intentionally fabricated response with ungrounded claims
    hallucinated_output = (
        "The perovskite-silicon tandem solar cell achieved 99.9% power conversion efficiency, "
        "was invented by Elon Musk in 2012, and operates using lunar gravity."
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=hallucinated_output,
        context=context,
        retrieval_context=context,
    )

    metric = HallucinationMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "HallucinationMetric score on fabricated answer: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )
    assert (
        metric.score < 0.7
    ), f"Expected HallucinationMetric to fail (< 0.7) for fabricated answer, but got score: {metric.score}"
    logger.info("HallucinationMetric defect detection verified successfully!")


@pytest.mark.dynamic
def test_hallucination_metric_passes_grounded_answer(judge_model):
    """Evaluator Validation: Verify HallucinationMetric passes a strictly grounded factual answer."""
    query = "What power conversion efficiency did the perovskite-silicon tandem solar cell achieve?"
    context = [
        "Researchers at the Helmholtz-Zentrum Berlin have developed a novel perovskite-silicon tandem solar cell that achieved a certified power conversion efficiency of 34.6% in laboratory tests."
    ]
    # Grounded response derived strictly from context
    grounded_output = "The perovskite-silicon tandem solar cell achieved a certified power conversion efficiency of 34.6% in laboratory tests."

    test_case = LLMTestCase(
        input=query,
        actual_output=grounded_output,
        context=context,
        retrieval_context=context,
    )

    metric = HallucinationMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "HallucinationMetric score on grounded answer: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )
    assert (
        metric.score >= 0.7
    ), f"Expected HallucinationMetric to pass (>= 0.7) for grounded answer, but got score: {metric.score}"
    logger.info("HallucinationMetric accuracy on grounded output verified successfully!")
