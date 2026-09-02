"""System Under Test (SUT) evaluation testing summarization quality using SummarizationMetric."""

import logging

from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_summarization_dynamic(judge_model, test_data_loader, response_generator):
    """SUT Core Test: Evaluate summarization truthfulness and keypoint coverage using DeepEval SummarizationMetric."""
    test_data = test_data_loader("summarization.json")

    logger.info("Starting summarization dynamic test")
    logger.info("Input Document:\n%s", test_data["input"])

    # Generate the actual summary output using LLM
    logger.info("Generating LLM summary...")
    system_instruction = (
        "Summarize the provided text accurately, capturing main concepts such as AI definition, benchmarking obsolescence, and modern domains."
    )
    actual_output = response_generator(test_data["input"], system_instruction=system_instruction)

    logger.info("Generated Summary:\n%s", actual_output)

    summarization_metric = SummarizationMetric(
        threshold=test_data.get("threshold", 0.6),
        model=judge_model,
    )

    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output,
    )

    summarization_metric.measure(test_case)

    logger.info(
        "SummarizationMetric score: %.2f (Reason: %s)",
        summarization_metric.score,
        summarization_metric.reason,
    )

    assert (
        summarization_metric.score >= 0.6
    ), f"Expected SummarizationMetric score >= 0.6, but got: {summarization_metric.score}"

    logger.info("Summarization dynamic test passed!")
