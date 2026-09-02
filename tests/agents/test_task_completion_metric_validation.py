"""Evaluator Validation test for DeepEval TaskCompletionMetric task completion & defect detection."""

import logging

from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_task_completion_metric_good_case(judge_model):
    """Evaluator Validation (GOOD Case): Verify TaskCompletionMetric gives a high score (>= 0.7) when task is fully completed."""
    query = "What is the weather in Tokyo and convert the temperature to Fahrenheit."
    actual_output = "The current temperature in Tokyo is 25°C, which equals 77°F with sunny skies."

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
    )
    test_case._trace_dict = {
        "name": "MinimalAgent",
        "type": "agent",
        "input": {"input": query},
        "children": [
            {
                "name": "get_weather",
                "type": "tool",
                "input": {"inputParameters": {"location": "Tokyo"}},
                "output": "The current temperature in Tokyo is 25°C with sunny skies.",
                "children": [],
            },
            {
                "name": "calculator",
                "type": "tool",
                "input": {"inputParameters": {"expression": "(25 * 9/5) + 32"}},
                "output": "77.0",
                "children": [],
            },
        ],
    }

    metric = TaskCompletionMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "TaskCompletionMetric score on fully completed task (GOOD Case): %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected TaskCompletionMetric score >= 0.7 for GOOD case, but got: {metric.score}"

    logger.info("TaskCompletionMetric GOOD case verified successfully!")


@pytest.mark.dynamic
def test_task_completion_metric_detects_incomplete_task(judge_model):
    """Evaluator Validation (BAD Case): Verify TaskCompletionMetric detects incomplete task execution and scores < 0.7."""
    query = "What is the weather in Tokyo and convert the temperature to Fahrenheit."
    incomplete_output = "The current temperature in Tokyo is 25°C with sunny skies."

    test_case = LLMTestCase(
        input=query,
        actual_output=incomplete_output,
    )
    # Controlled defective trace: weather tool called, but Fahrenheit conversion step missing
    test_case._trace_dict = {
        "name": "MinimalAgent",
        "type": "agent",
        "input": {"input": query},
        "children": [
            {
                "name": "get_weather",
                "type": "tool",
                "input": {"inputParameters": {"location": "Tokyo"}},
                "output": "The current temperature in Tokyo is 25°C with sunny skies.",
                "children": [],
            },
        ],
    }

    metric = TaskCompletionMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "TaskCompletionMetric score on incomplete task defect (BAD Case): %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score < 0.7
    ), f"Expected TaskCompletionMetric score < 0.7 for BAD case, but got: {metric.score}"

    logger.info("TaskCompletionMetric BAD case defect detection verified successfully!")

