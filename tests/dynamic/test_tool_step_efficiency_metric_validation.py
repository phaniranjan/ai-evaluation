"""Evaluator Validation test for DeepEval StepEfficiencyMetric step inefficiency defects."""

import logging

from deepeval.metrics import StepEfficiencyMetric
from deepeval.test_case import LLMTestCase
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_step_efficiency_metric_detects_redundant_steps(judge_model):
    """Evaluator Validation: Verify StepEfficiencyMetric detects redundant/superfluous steps and scores < 0.7."""
    query = "What is the weather in Tokyo and convert the temperature to Fahrenheit."

    test_case = LLMTestCase(
        input=query,
        actual_output="The current temperature in Tokyo is 25°C (77°F).",
    )

    # Controlled inefficient trajectory: 6 steps with redundant tool calls and unrelated cities
    test_case._trace_dict = {
        "name": "MinimalAgent",
        "type": "agent",
        "input": {"input": query},
        "children": [
            {
                "name": "get_weather",
                "type": "tool",
                "input": {"inputParameters": {"location": "London"}},
                "output": "15°C",
                "children": [],
            },
            {
                "name": "get_time",
                "type": "tool",
                "input": {"inputParameters": {"location": "Paris"}},
                "output": "18:00",
                "children": [],
            },
            {
                "name": "get_weather",
                "type": "tool",
                "input": {"inputParameters": {"location": "Tokyo"}},
                "output": "25°C",
                "children": [],
            },
            {
                "name": "calculator",
                "type": "tool",
                "input": {"inputParameters": {"expression": "25 * 9/5 + 32"}},
                "output": "77.0",
                "children": [],
            },
            {
                "name": "calculator",
                "type": "tool",
                "input": {"inputParameters": {"expression": "25 * 9/5 + 32"}},
                "output": "77.0",
                "children": [],
            },
            {
                "name": "get_time",
                "type": "tool",
                "input": {"inputParameters": {"location": "Tokyo"}},
                "output": "02:00",
                "children": [],
            },
        ],
    }

    metric = StepEfficiencyMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "StepEfficiencyMetric score on redundant step trajectory defect: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    # Assert that evaluator detects step inefficiency defect and scores below 0.7 threshold
    assert (
        metric.score < 0.7
    ), f"Expected StepEfficiencyMetric to fail (< 0.7) for redundant steps, but got score: {metric.score}"

    logger.info("StepEfficiencyMetric step inefficiency defect detection verified successfully!")

