"""Evaluator Validation test for DeepEval ToolCorrectnessMetric trajectory ordering and execution defects."""

import logging

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_tool_trajectory_metric_detects_out_of_order_execution(judge_model):
    """Evaluator Validation: Verify ToolCorrectnessMetric detects out-of-order tool trajectory defects and scores < 0.7."""
    query = "What is the weather in Tokyo and convert the temperature to Fahrenheit."

    # Controlled bad trajectory: calculator called FIRST before get_weather
    controlled_bad_trajectory = [
        ToolCall(
            name="calculator",
            input_parameters={"expression": "25 * 9 / 5 + 32"},
            output="77.0",
        ),
        ToolCall(
            name="get_weather",
            input_parameters={"location": "Tokyo"},
            output="The current temperature in Tokyo is 25°C with sunny skies.",
        ),
    ]

    expected_trajectory = [
        ToolCall(
            name="get_weather",
            input_parameters={"location": "Tokyo"},
            output="The current temperature in Tokyo is 25°C with sunny skies.",
        ),
        ToolCall(
            name="calculator",
            input_parameters={"expression": "25 * 9 / 5 + 32"},
            output="77.0",
        ),
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output="The temperature in Tokyo is 25°C (77°F).",
        tools_called=controlled_bad_trajectory,
        expected_tools=expected_trajectory,
    )

    metric = ToolCorrectnessMetric(
        available_tools=[
            ToolCall(name="get_weather"),
            ToolCall(name="calculator"),
            ToolCall(name="get_time"),
        ],
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS, ToolCallParams.OUTPUT],
        should_consider_ordering=True,
        threshold=0.7,
        model=judge_model,
    )

    metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric score on out-of-order trajectory defect: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    # Assert that evaluator detects trajectory ordering failure and scores below 0.7 threshold
    assert (
        metric.score < 0.7
    ), f"Expected ToolCorrectnessMetric to fail (< 0.7) for out-of-order trajectory, but got score: {metric.score}"

    logger.info("ToolCorrectnessMetric trajectory ordering defect detection verified successfully!")

