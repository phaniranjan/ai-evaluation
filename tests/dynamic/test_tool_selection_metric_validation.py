"""Evaluator Validation test for DeepEval ToolCorrectnessMetric tool selection."""

import logging

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_tool_selection_metric_detects_wrong_tool(judge_model):
    """Evaluator Validation: Verify ToolCorrectnessMetric detects incorrect tool selection and produces score < 0.7."""
    query = "What is the current weather and temperature in Tokyo?"

    # Forced incorrect tool selection (calculator instead of get_weather)
    forced_incorrect_tools = [ToolCall(name="calculator")]
    expected_tools = [ToolCall(name="get_weather")]

    test_case = LLMTestCase(
        input=query,
        actual_output="I used the calculator tool to check.",
        tools_called=forced_incorrect_tools,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric score on wrong tool selection: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    # Assert that evaluator detects tool selection failure and scores below 0.7 threshold
    assert (
        metric.score < 0.7
    ), f"Expected ToolCorrectnessMetric to fail (< 0.7) for wrong tool selection, but got score: {metric.score}"

    logger.info("ToolCorrectnessMetric tool selection defect detection verified successfully!")
