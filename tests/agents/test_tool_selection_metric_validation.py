"""Evaluator Validation test for DeepEval ToolCorrectnessMetric tool selection across multiple tools."""

import logging

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_tool_selection_metric_detects_wrong_tool(judge_model):
    """Evaluator Validation: Verify ToolCorrectnessMetric detects incorrect tool selection when multiple tools are available."""
    query = "What is the weather in Tokyo?"

    # Available tools in system: get_weather, calculator, get_time
    # Forced defect: selected calculator instead of get_weather
    forced_incorrect_tools = [ToolCall(name="calculator")]
    expected_tools = [ToolCall(name="get_weather")]

    test_case = LLMTestCase(
        input=query,
        actual_output="I selected the calculator tool.",
        tools_called=forced_incorrect_tools,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric score on wrong tool selection defect: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    # Assert that evaluator detects tool selection failure and scores below 0.7 threshold
    assert (
        metric.score < 0.7
    ), f"Expected ToolCorrectnessMetric to fail (< 0.7) for wrong tool selection, but got score: {metric.score}"

    logger.info("ToolCorrectnessMetric tool selection defect detection verified successfully!")
