"""Evaluator Validation test for DeepEval ArgumentCorrectnessMetric tool argument accuracy & defect detection."""

import logging

from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_argument_correctness_metric_valid_arguments(judge_model):
    """Evaluator Validation (GOOD Case): Verify ArgumentCorrectnessMetric gives score 1.00 when tool arguments accurately match query intent."""
    query = "What is the weather in Tokyo?"
    actual_output = "The current temperature in Tokyo is 25°C with sunny skies."

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        tools_called=[
            ToolCall(
                name="get_weather",
                input_parameters={"location": "Tokyo"},
                output="25°C",
            )
        ],
    )

    metric = ArgumentCorrectnessMetric(threshold=0.5, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ArgumentCorrectnessMetric score on valid arguments (GOOD Case): %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ArgumentCorrectnessMetric score >= 0.7 for valid arguments, but got: {metric.score}"

    logger.info("ArgumentCorrectnessMetric valid arguments verified successfully!")


@pytest.mark.dynamic
def test_argument_correctness_metric_detects_wrong_location_argument_defect(judge_model):
    """Evaluator Validation (BAD Case): Verify ArgumentCorrectnessMetric detects mismatched location parameter (Tokyo vs Paris) and scores < 0.5."""
    query = "What is the weather in Tokyo?"
    actual_output = "The current temperature in Paris is 18°C."

    # Controlled defect: User asked for Tokyo, but tool received location="Paris"
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        tools_called=[
            ToolCall(
                name="get_weather",
                input_parameters={"location": "Paris"},
                output="18°C",
            )
        ],
    )

    metric = ArgumentCorrectnessMetric(threshold=0.5, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ArgumentCorrectnessMetric score on wrong location argument defect (BAD Case): %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score < 0.5
    ), f"Expected ArgumentCorrectnessMetric score < 0.5 for wrong location argument defect, but got: {metric.score}"

    logger.info("ArgumentCorrectnessMetric wrong location argument defect detection verified successfully!")


@pytest.mark.dynamic
def test_argument_correctness_metric_detects_wrong_math_expression_argument_defect(judge_model):
    """Evaluator Validation (BAD Case): Verify ArgumentCorrectnessMetric detects mismatched math expression parameter (100+500 vs 25*9/5+32) and scores < 0.5."""
    query = "Calculate 25 * 9/5 + 32"
    actual_output = "The result is 600."

    # Controlled defect: User asked to calculate (25 * 9/5) + 32, but tool received expression="100 + 500"
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        tools_called=[
            ToolCall(
                name="calculator",
                input_parameters={"expression": "100 + 500"},
                output="600",
            )
        ],
    )

    metric = ArgumentCorrectnessMetric(threshold=0.5, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ArgumentCorrectnessMetric score on wrong math expression argument defect (BAD Case): %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score < 0.5
    ), f"Expected ArgumentCorrectnessMetric score < 0.5 for wrong math expression defect, but got: {metric.score}"

    logger.info("ArgumentCorrectnessMetric wrong math expression defect detection verified successfully!")

