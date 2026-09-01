"""System Under Test (SUT) evaluation testing MinimalAgent tool selection capabilities."""

import logging

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_minimal_agent_tool_selection(judge_model, minimal_agent):
    """SUT Agent Test: Verify MinimalAgent correctly selects the get_weather tool for a weather query."""
    query = "What is the current weather and temperature in Tokyo?"

    # Execute MinimalAgent to capture Gemini's actual tool selection
    result = minimal_agent.run(query)

    actual_tools_called = result["tools_called"]
    actual_answer = result["answer"]

    logger.info("MinimalAgent query: %s", query)
    logger.info("MinimalAgent actual tools called: %s", actual_tools_called)
    logger.info("MinimalAgent actual answer text: %s", actual_answer)

    expected_tools = [ToolCall(name="get_weather")]

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_answer,
        tools_called=actual_tools_called,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric score on MinimalAgent: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ToolCorrectnessMetric to pass (>= 0.7), but got score: {metric.score}"
    logger.info("MinimalAgent tool selection SUT evaluation passed successfully!")
