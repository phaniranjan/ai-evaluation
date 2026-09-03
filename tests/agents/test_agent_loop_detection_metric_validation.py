"""Evaluator Validation test for DeepEval AgentLoopDetectionMetric loop & repetition defect detection."""

import logging

from deepeval.metrics import AgentLoopDetectionMetric
from deepeval.test_case import LLMTestCase
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_agent_loop_detection_metric_healthy_trajectory():
    """Evaluator Validation (GOOD Case): Verify AgentLoopDetectionMetric gives score 1.00 for a healthy, non-repetitive trajectory."""
    query = "What is the weather in Tokyo and convert the temperature to Fahrenheit."
    actual_output = "The current temperature in Tokyo is 25°C (77°F)."

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
    )
    # Controlled healthy trace: 2 distinct tool calls executed sequentially without repetition
    test_case._trace_dict = {
        "name": "MinimalAgent",
        "type": "agent",
        "input": {"input": query},
        "children": [
            {
                "name": "get_weather",
                "type": "tool",
                "input": {"location": "Tokyo"},
                "output": "25°C",
                "children": [],
            },
            {
                "name": "calculator",
                "type": "tool",
                "input": {"expression": "(25 * 9/5) + 32"},
                "output": "77.0",
                "children": [],
            },
        ],
    }

    metric = AgentLoopDetectionMetric(
        threshold=0.5, repetition_threshold=3, strict_mode=True
    )
    metric.measure(test_case)

    logger.info(
        "AgentLoopDetectionMetric score on healthy trajectory (GOOD Case): %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected AgentLoopDetectionMetric score >= 0.7 for healthy trajectory, but got: {metric.score}"

    logger.info("AgentLoopDetectionMetric healthy trajectory verified successfully!")


@pytest.mark.dynamic
def test_agent_loop_detection_metric_detects_infinite_loop_defect():
    """Evaluator Validation (BAD Case): Verify AgentLoopDetectionMetric detects infinite tool loop defect and scores < 0.5."""
    query = "What is the weather in Atlantis?"
    actual_output = "Error: Repeatedly failed to retrieve weather for Atlantis."

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
    )
    # Controlled loop defect trace: 6 identical failing tool calls executed sequentially
    test_case._trace_dict = {
        "name": "MinimalAgent",
        "type": "agent",
        "input": {"input": query},
        "children": [
            {
                "name": "get_weather",
                "type": "tool",
                "input": {"location": "Atlantis"},
                "output": "Error 404: Location 'Atlantis' not found in database.",
                "children": [],
            }
            for _ in range(6)
        ],
    }

    metric = AgentLoopDetectionMetric(
        threshold=0.5, repetition_threshold=3, strict_mode=True
    )
    metric.measure(test_case)

    logger.info(
        "AgentLoopDetectionMetric score on infinite loop defect (BAD Case): %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score < 0.5
    ), f"Expected AgentLoopDetectionMetric score < 0.5 for infinite loop defect, but got: {metric.score}"

    logger.info("AgentLoopDetectionMetric infinite loop defect detection verified successfully!")

