"""System Under Test (SUT) evaluation testing MinimalAgent tool selection across multiple tools."""

import logging

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "user_query,expected_tool_name",
    [
        ("What's the weather in Tokyo?", "get_weather"),
        ("What is 125 * 37?", "calculator"),
        ("What time is it in Tokyo?", "get_time"),
    ],
)
@pytest.mark.dynamic
def test_minimal_agent_multi_tool_selection(
    judge_model, minimal_agent, user_query, expected_tool_name
):
    """SUT Agent Test: Verify MinimalAgent correctly selects the appropriate tool among multiple declared tools."""
    logger.info("Executing MinimalAgent multi-tool selection test for query: %s", user_query)

    # Execute MinimalAgent to capture Gemini's actual tool selection among 3 tools
    result = minimal_agent.run(user_query)

    actual_tools_called = result["tools_called"]
    actual_answer = result["answer"]

    logger.info("Query: %s | Expected Tool: %s", user_query, expected_tool_name)
    logger.info("Actual tools called: %s", actual_tools_called)

    expected_tools = [ToolCall(name=expected_tool_name)]

    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_answer,
        tools_called=actual_tools_called,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric score for query '%s': %.2f (Reason: %s)",
        user_query,
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ToolCorrectnessMetric to pass (>= 0.7) for '{user_query}', but got score: {metric.score}"

    logger.info("Tool selection evaluation passed successfully for tool: %s", expected_tool_name)
